local ExpandRepeated = require('sconce.ExpandRepeated')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_expand_repeated_forward()
    local input = torch.Tensor{1, 2}
    local expand_repeated = ExpandRepeated.new(5, 1, 1)
    local actual = expand_repeated:forward(input)
    local expected = torch.Tensor{1, 2, 1, 2, 1}
    tester:eq(actual, expected)
  end

  function suite.test_expand_repeated_backward()
    local input = torch.Tensor{1, 2}
    local grad_output = torch.Tensor{1, 4, 2, 5, 3}
    local expand_repeated = ExpandRepeated.new(5, 1, 1)
    local actual = expand_repeated:backward(input, grad_output)
    local expected = torch.Tensor{6, 9}
    tester:eq(actual, expected)
  end

  tester:add(suite)

  return suite
end
