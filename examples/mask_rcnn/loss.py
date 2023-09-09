# RCNN-specific loss functions

from models.mask_rcnn import *
from tinygrad.tensor import Tensor
from tinygrad.tensor import dtypes
import numpy as np
from typing import List, Callable, Tuple
from torch.nn import functional as F

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications

def test_boxlist_iou():
  a = boxlist_iou(BoxList(Tensor([[0, 0, 10, 10]]), image_size = (50, 50)), BoxList(Tensor([[0, 0, 5, 5]]), image_size = (50, 50)))
  assert all(((a == .25)[0]).numpy())


def boxlist_iou(boxlist1: BoxList, boxlist2: BoxList) -> Tensor:
  # Compute the intersection over union of two set of boxes.
  assert boxlist1.size == boxlist2.size, f"boxlists should have same image size, got {boxlist1}, {boxlist2}"
  N, M = len(boxlist1), len(boxlist2)
  area1, area2 = boxlist1.area(), boxlist2.area()
  box1, box2 = boxlist1.bbox, boxlist2.bbox
  lt = Tensor.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
  rb = Tensor.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
  wh = (rb - lt).maximum(0)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

  iou = inter / (area1[:, None] + area2 - inter)
  return iou


def test_match_eval():
  fn1, _ = make_match_fn(0.7, 0.4)
  match_quality_matrix = Tensor([[0.9, 0.7, 0.8, 0.9, .1],
                                [0.1, 0.5, 0.1, 0.2, .1],
                                [0.1, 0.2, 0.85, 0.3, .1],
                                [0.1, 0.9, 0.2, 0.3, .3],
                                [0.1, 0.9, 0.2, 0.8, .1]])
  a = fn1(match_quality_matrix)
  assert all(((a == Tensor([0, 4, 2, 0, -1]))).numpy())

def test_low_qual_match():
  matches = Tensor([
    [.3, .4, .1], #gt 1
    [.4, .5, .2],
    [.5, .8, .7],
    [.6, .1, .0]
  ])
  hq_fn, lq_fn = make_match_fn(.7, .3)
  first_pass = hq_fn(matches)
  second_pass = (first_pass == -1).where(lq_fn(matches), first_pass) # merges in low quality matches
  assert all(((second_pass == Tensor([3, 2, 2]))).numpy())
  # tests where matches are either hq or negative (weak signals filtered)
  low_signal = ((matches >= .3) * (matches <= .7)).sum(axis=0) == matches.shape[0]
  res = low_signal.where(-2, first_pass)
  assert all(res == Tensor([-2, 2, 2]).numpy())

def make_match_fn(high: float, low: float) -> Callable[[Tensor], Tensor]:
  # for the tensor of M*N
  # where M is the index of the gt and N is the prediction's quality for that gt
  # returns a tensor of N length, where N[i] is the gt that best matched this prediction
  # N[i] is a negative value if there is no match
  def hq_match_fn(preds: Tensor) -> Tensor:
    assert preds.numel() > 0, "must be scoring something"
    # drops lq+mid matches early
    hq = (preds >= high) * preds
    max_vals = hq.max(axis=0)
    max_vals = (max_vals == 0).where(-1, max_vals) # -1 when pred == 0 for all gt
    best_matches = (hq == max_vals).float()
    best_gt = (best_matches * Tensor.ones_like(best_matches).cumsum(axis=0)).max(axis=0)
    return best_gt - 1
  
  # Returns matches that were greater than low and less than high
  # TODO dry
  def lq_match_fn(preds: Tensor) -> Tensor:
    assert preds.numel() > 0, "must be scoring something"
    # drops lq+mid matches early
    lq = (preds < high) * (preds >= low) * preds
    max_vals = lq.max(axis=0)
    max_vals = (max_vals == 0).where(-1, max_vals) # -1 when pred == 0 for all gt
    best_matches = (lq == max_vals).float()
    best_gt = (best_matches * Tensor.ones_like(best_matches).cumsum(axis=0)).max(axis=0)
    return best_gt - 1

  return hq_match_fn, lq_match_fn

def test_rind():
  x = rind(Tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1]).numpy(), 3)
  assert x.ndim == 1
  assert x.shape[0] == 3
  assert np.isin(x, [0, 1, 2, 3, 8, 9]).all()

# TODO perf
# returns random indices of a mask
def rind(mask: np.ndarray, take: int) -> Tensor:
  assert mask.ndim == 1 and mask.shape[0] >= take
  masked = (np.arange(mask.shape[0]) * mask)[mask.astype(bool)]
  stacked = np.stack([masked,np.random.rand(masked.shape[0])],axis=0)
  return stacked[0, stacked[1].argsort()[:take]]

def test_balanced_sampler():
  fn1 = make_balanced_sampler_fn(10, 0.5)
  t1 = Tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
  a1 = np.arange(t1.shape[0])
  a, b = fn1([t1])
  assert np.isin(a[0], t1.numpy() * a1).all()
  assert np.isin(b[0], (t1 == 0).numpy() * a1).all()

# returns random mask of positive and negative examples
def make_balanced_sampler_fn(batch_size_per_image: int, positive_fraction: float) -> Callable[[Tensor], Tuple[List[Tensor], List[Tensor]]]:
  def sampler_fn(image_matches: List[Tensor]) -> (Tensor, Tensor):
    pos_inds = []
    neg_inds = []
    for matches in image_matches:
      # expected that positive samples are amped to 1
      positive, negative = matches == 1, matches == 0 
      num_pos = int(batch_size_per_image * positive_fraction)
      
      # protect against not enough positive examples
      pos_numel, neg_numel = positive.sum().numpy().item(), negative.sum().numpy().item()
      num_pos = int(min(pos_numel, num_pos))
      num_neg = int(min(neg_numel, batch_size_per_image - num_pos))
      
      # option .. return a mask or return gather indices, which is more efficient?
      pos_inds.append(rind(positive.numpy(), num_pos).astype(int))
      neg_inds.append(rind(negative.numpy(), num_neg).astype(int))

    return pos_inds, neg_inds
  return sampler_fn

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_idxs: Tensor) -> Tensor:
    labels_per_image = matched_idxs >= 0
    return labels_per_image

def permute_and_flatten(layer, N, A, C, H, W):
  layer = layer.view(N, -1, C, H, W)
  layer = layer.permute(0, 3, 4, 1, 2)
  layer = layer.reshape(N, -1, C)
  return layer

def concat_box_prediction_layers(box_cls, box_regression):
  box_cls_flattened = []
  box_regression_flattened = []
  # for each feature level, permute the outputs to make them be in the
  # same format as the labels. Note that the labels are computed for
  # all feature levels concatenated, so we keep the same representation
  # for the objectness and the box_regression
  for box_cls_per_level, box_regression_per_level in zip(
      box_cls, box_regression
  ):
      N, AxC, H, W = box_cls_per_level.shape
      Ax4 = box_regression_per_level.shape[1]
      A = Ax4 // 4
      C = AxC // A
      box_cls_per_level = permute_and_flatten(
          box_cls_per_level, N, A, C, H, W
      )
      box_cls_flattened.append(box_cls_per_level)

      box_regression_per_level = permute_and_flatten(
          box_regression_per_level, N, A, 4, H, W
      )
      box_regression_flattened.append(box_regression_per_level)
  # concatenate on the first dimension (representing the feature levels), to
  # take into account the way the labels were generated (with all feature maps
  # being concatenated as well)
  box_cls = Tensor.cat(box_cls_flattened, dim=1).reshape(-1, C)
  box_regression = Tensor.cat(box_regression_flattened, dim=1).reshape(-1, 4)
  return box_cls, box_regression

# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
  """
  very similar to the smooth_l1_loss from pytorch, but with
  the extra beta parameter
  """
  n = Tensor.abs(input - target)
  cond = n < beta
  loss = Tensor.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
  if size_average:
      return loss.mean()
  return loss.sum()

def test_match_targets_to_anchors():
  anchors = BoxList(Tensor([[0, 0, 10, 10], [0, 0, 5, 5]]), image_size = (50, 50)) # preds
  targets = BoxList(Tensor([[0, 0, 5, 5], [0, 0, 10, 10]]), image_size = (50, 50))
  hq_fn, _ = make_match_fn(0.7, 0.4)
  loss = RPNLossComputation(hq_fn, None, None, generate_rpn_labels)
  matched_targets, _ = loss.match_targets_to_anchors(anchors, targets)
  result = Tensor([[0, 0, 10, 10], [0, 0, 5, 5]])
  assert (matched_targets.bbox == result).numpy().all()

def test_prepare_targets():
  hq_fn, _ = make_match_fn(0.7, 0.4)
  sampler = make_balanced_sampler_fn(10, 0.5)
  rpn = RPNLossComputation(hq_fn, sampler, BoxCoder(weights=(1.0, 1.0, 1.0, 1.0)), generate_rpn_labels)
  labels,regression_targets = rpn.prepare_targets(
    [BoxList(Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [12, 12, 14, 14]]), image_size = (50, 50))],
    [BoxList(Tensor([[0, 0, 5, 5], [0, 0, 10, 10]]), image_size = (50, 50))]
  )
  assert (labels[0] == Tensor([1, 1, 0])).numpy().all() # good matches, fg, bad matches, bg
  assert np.allclose(
    rpn.box_coder.decode(
      regression_targets[0],
      Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [12, 12, 14, 14]])
    ).numpy(),
    Tensor([[0, 0, 10, 10], [0, 0, 5, 5], [0, 0, 5, 5]]).numpy(),
    atol=1e-6 ## TODO currently drift is 1e-7, why?
  )

def test_loss():
  hq_fn, _ = make_match_fn(0.7, 0.4)
  sampler = make_balanced_sampler_fn(10, 0.5)
  rpn = RPNLossComputation(hq_fn, sampler, BoxCoder(weights=(1.0, 1.0, 1.0, 1.0)), generate_rpn_labels)
  rpn(
    anchors=[BoxList([
      [10, 10, 20, 20], [0, 0, 5, 5], [12, 12, 18, 18],
      [15, 10, 25, 18], [22, 5, 30, 25], [12, 12, 16, 16]
    ], image_size=(50, 50))],
    objectness=[Tensor([0.3, 0.8, 0.1, 0.5, 0.7, 0.2])],
    box_regression=[Tensor([
      [0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 2, 2],
      [0, 0, 1, 2], [0, 0, 3, 1], [1, 0, 2, 1]
    ])],
    targets=[BoxList([
      [0, 0, 5, 5], [5, 5, 10, 10], [15, 15, 20, 20],
      [25, 25, 30, 30]
    ], image_size=(50, 50))]
  )

# one way this differs from reference is it doesn't rely on boxlist mutables
class RPNLossComputation:
  def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
              generate_labels_func):
    """
    Arguments: 
        proposal_matcher (Matcher)
        fg_bg_sampler (BalancedPositiveNegativeSampler)
        box_coder (BoxCoder)
    """
    # self.target_preparator = target_preparator
    self.proposal_matcher = proposal_matcher
    self.fg_bg_sampler = fg_bg_sampler
    self.box_coder = box_coder
    self.generate_labels_func = generate_labels_func
    self.discard_cases = ['not_visibility', 'between_thresholds']

  def match_targets_to_anchors(self, anchors: BoxList, targets: BoxList):
    match_quality_matrix = boxlist_iou(targets, anchors)
    matched_idxs = self.proposal_matcher(match_quality_matrix)
    matched_targets = targets[matched_idxs.maximum(0)] # drop negatives
    return matched_targets, matched_idxs

  def prepare_targets(self, anchors: List[BoxList], targets: List[BoxList]):
    labels = []
    regression_targets = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
      matched_targets, matched_idxs = self.match_targets_to_anchors(
          anchors_per_image, targets_per_image
      )

      # TODO this has fp errors
      regression_targets_per_image = self.box_coder.encode(
          matched_targets.bbox, anchors_per_image.bbox
      )

      # all matches become 1 (roi head) (.7 and above amplified)
      labels_per_image = self.generate_labels_func(matched_idxs)
      labels_per_image = labels_per_image.cast(dtype=dtypes.float32)

      # negative samples are labeled 0
      labels_per_image = (matched_idxs == -1).where(0, labels_per_image)

      # TODO: discard anchors that go out of the boundaries of the image
      # labels_per_image[~anchors_per_image.get_field("visibility")] = -1

      # discards weak signals (when lq matches is False), -1 is ignored by fg_bg_sampler
      labels_per_image = (matched_idxs == -2).where(-1, labels_per_image)

      labels.append(labels_per_image)
      regression_targets.append(regression_targets_per_image)

    return labels, regression_targets


  def __call__(self, anchors: list[BoxList], objectness: list[Tensor],
               box_regression: list[Tensor], targets: list[BoxList]):
    """
    Arguments:
        anchors (list[BoxList])
        objectness (list[Tensor])
        box_regression (list[Tensor])
        targets (list[BoxList])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor
    """
    anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors] # what this doin
    labels, regression_targets = self.prepare_targets(anchors, targets)
    sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
    sampled_pos_inds = Tensor.nonzero(Tensor.cat(sampled_pos_inds, dim=0)).squeeze(1)
    sampled_neg_inds = Tensor.nonzero(Tensor.cat(sampled_neg_inds, dim=0)).squeeze(1)

    sampled_inds = Tensor.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)

    objectness = objectness.squeeze()

    labels = Tensor.cat(labels, dim=0)
    regression_targets = Tensor.cat(regression_targets, dim=0)

    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=False,
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss

if __name__ == "__main__":
  #PLAYGROUND
  data = Tensor([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9],
               [10, 11, 12]])
  idx = Tensor([0, 2, 1, 1]).reshape(4, 1)
  result = data.gather(idx, dim=1)
  test_boxlist_iou()
  test_match_eval()
  test_low_qual_match()
  test_rind()
  test_balanced_sampler()
  test_match_targets_to_anchors()
  test_prepare_targets()
  test_loss()
  
  # ind = Tensor.arange(mask.shape[0])
  # nz = mask.sum().numpy().item()
  # mask = mask * ind
  # idx = mask.numpy().argsort()[-int(nz):]