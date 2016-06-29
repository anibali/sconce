local Sideload = require('sconce.Sideload')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_sideload_forward()
    local get_tensor = function()
      return torch.Tensor{2, 2}
    end
    local input = torch.Tensor{1, 1}
    local sideload = Sideload.new(get_tensor)
    local actual = sideload:forward(input)
    local expected = {torch.Tensor{1, 1}, torch.Tensor{2, 2}}
    tester:eq(actual, expected)
  end

  function suite.test_sideload_backward()
    local get_tensor = function()
      return torch.Tensor{2, 2}
    end
    local input = torch.Tensor{1, 1}
    local sideload = Sideload.new(get_tensor)
    local grad_output = {torch.Tensor{3, 3}, torch.Tensor{4, 4}}
    local actual = sideload:backward(input, grad_output)
    local expected = torch.Tensor{3, 3}
    tester:eq(actual, expected)
  end

  tester:add(suite)

  return suite
end
