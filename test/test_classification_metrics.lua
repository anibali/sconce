local ClassificationMetrics = require('sconce.ClassificationMetrics')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_classification_metrics_top_k_accuracy()
    local outputs = torch.Tensor{
      {0.8, 0.1, 0.1},
      {0.5, 0.4, 0.1},
      {0.3, 0.3, 0.4}
    }
    local targets = torch.LongTensor{1, 2, 3}
    local cm = ClassificationMetrics.new(outputs, targets, 3)
    local top1_acc, top2_acc = cm:top_k_accuracy(1, 2)
    tester:eq(top1_acc, 2 / 3, 0.0001)
    tester:eq(top2_acc, 3 / 3, 0.0001)
  end

  tester:add(suite)

  return suite
end
