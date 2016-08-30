local AddGaussianNoise = require('sconce.AddGaussianNoise')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_add_gaussian_noise_forward()
    local input = torch.Tensor{1, 2, 3, 4}
    local add_noise = AddGaussianNoise.new(0.1)
    torch.manualSeed(1234)
    local actual = add_noise:forward(input)
    local expected = torch.Tensor{1.0422, 2.1095, 2.8672, 3.8719}
    tester:eq(actual, expected, 0.001)
  end

  function suite.test_add_gaussian_noise_backward()
    local input = torch.Tensor{1, 2, 3, 4}
    local grad_output = torch.Tensor{0.1, 0.2, 0.3, 0.4}
    local add_noise = AddGaussianNoise.new(0.1)
    local actual = add_noise:backward(input, grad_output)
    local expected = grad_output:clone()
    tester:eq(actual, expected)
  end

  tester:add(suite)

  return suite
end
