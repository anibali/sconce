local torch = require('torch')

local ClassificationMetrics =
  require('sconce.ns').class('sconce.ClassificationMetrics')

-- outputs: Model outputs (output of softmax for each example)
-- targets: Targets (single class label per example)
function ClassificationMetrics:__init(outputs, targets, n_thresholds)
  self.outputs = outputs
  self.targets = targets
  self.n_thresholds = n_thresholds or 101
  self.n_examples = outputs:size(1)
  self.n_classes = outputs:size(2)
end

-- Accepts one or more k values, returns corresponding top-k accuracies
function ClassificationMetrics:top_k_accuracy(...)
  local ks = {...}
  assert(#ks > 0)

  local accuracies = {}

  for i = 1,self.n_examples do
    local _ignore, sorted_indices = self.outputs[i]:sort(1, true)

    local matches = sorted_indices:eq(self.targets:narrow(1, i, 1):expandAs(sorted_indices))

    for k_index,k in ipairs(ks) do
      accuracies[k_index] = (accuracies[k_index] or 0) +
        (matches:narrow(1, 1, k):sum() / self.n_examples)
    end
  end

  return unpack(accuracies)
end

function ClassificationMetrics:assess_outputs()
  if self.simple_metrics then
    return self.simple_metrics
  end

  -- local thresholds = {}
  -- local probabilities = self.outputs:view(-1):sort()
  -- for i=1,self.n_thresholds do
  --   table.insert(thresholds, probabilities[i * (probabilities:size(1) / self.n_thresholds)])
  -- end
  local thresholds = torch.linspace(0, 1, self.n_thresholds):totable()

  thresholds[1] = thresholds[1] - 0.0001
  thresholds[#thresholds] = thresholds[#thresholds] + 0.0001

  local tps = torch.FloatTensor(self.n_thresholds, self.n_classes):zero()
  local fps = torch.FloatTensor(self.n_thresholds, self.n_classes):zero()
  local tns = torch.FloatTensor(self.n_thresholds, self.n_classes):zero()
  local fns = torch.FloatTensor(self.n_thresholds, self.n_classes):zero()

  local one_hot_targets = torch.ByteTensor(self.outputs:size()):zero()
  one_hot_targets:scatter(2, self.targets:long():view(-1, 1), 1)
  local not_one_hot_targets = one_hot_targets:eq(0)

  for threshold_index,threshold in ipairs(thresholds) do
    local thresholded_outputs = torch.ge(self.outputs, threshold)
    local not_thresholded_outputs = thresholded_outputs:eq(0)

    tps[threshold_index] = torch.cmul(one_hot_targets, thresholded_outputs):sum(1)
    fps[threshold_index] = torch.cmul(not_one_hot_targets, thresholded_outputs):sum(1)
    fns[threshold_index] = torch.cmul(one_hot_targets, not_thresholded_outputs):sum(1)
    tns[threshold_index] = torch.cmul(not_one_hot_targets, not_thresholded_outputs):sum(1)
  end

  self.simple_metrics = {
    thresholds = torch.FloatTensor(thresholds),
    true_positives = tps,
    false_positives = fps,
    true_negatives = tns,
    false_negatives = fns
  }

  return self.simple_metrics
end

-- Return dim: n_thresholds x n_classes
function ClassificationMetrics:precision()
  local metrics = self:assess_outputs()

  local test_positives = torch.add(metrics.true_positives, metrics.false_positives)
  local precisions = metrics.true_positives:clone()
  precisions:map(test_positives, function(x, y)
    if y == 0 then
      return 1
    else
      return x / y
    end
  end)

  return precisions
end

-- Return dim: n_thresholds x n_classes
function ClassificationMetrics:recall()
  local metrics = self:assess_outputs()

  local condition_positives = torch.add(metrics.true_positives, metrics.false_negatives)
  local recalls = metrics.true_positives:clone()
  recalls:map(condition_positives, function(x, y)
    if y == 0 then
      return 1
    else
      return x / y
    end
  end)

  return recalls
end

-- Returns the average precision for each class (approximated by taking the
-- area under a precision-recall curve)
--
-- Return dim: n_classes
function ClassificationMetrics:average_precision()
  local metrics = self:assess_outputs()

  local precisions = self:precision()
  local recalls = self:recall()

  -- Average precision = area under precision-recall curve
  local aps = torch.FloatTensor(self.n_classes):zero()

  for i = 1,(recalls:size(1) - 1) do
    local trapezium_area = torch.cmul(recalls[i] - recalls[i + 1],
      (torch.add(precisions[i], precisions[i + 1]) / 2))
    aps:add(trapezium_area)
  end

  return aps
end

return ClassificationMetrics
