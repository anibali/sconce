local WeightedBCECriterion = require('sconce.WeightedBCECriterion')



return function(tester)
  local function criterionJacobianTest(cri, input, target)
    local eps = 1e-6
    local _ = cri:forward(input, target)
    local dfdx = cri:backward(input, target)
    -- for each input perturbation, do central difference
    local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
    local input_s = input:storage()
    local centraldiff_dfdx_s = centraldiff_dfdx:storage()
    for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
    end

    -- compare centraldiff_dfdx with :backward()
    local err = (centraldiff_dfdx - dfdx):abs():max()
    tester:assertlt(err, 1e-5, 'error in difference between central difference and :backward')
  end

  local suite = torch.TestSuite()

  function suite.test_weighted_bce_forward()
    local input = torch.Tensor{0.2}
    local target = torch.Tensor{0}
    local criterion = WeightedBCECriterion.new(0.5)

    local actual = criterion:forward(input, target)
    local expected = nn.BCECriterion():forward(input, target)
    tester:eq(actual, expected, 1e-4)
  end

  function suite.test_weighted_bce_forward_pos_weight()
    local input = torch.Tensor{0.2}
    local target = torch.Tensor{0}
    local criterion = WeightedBCECriterion.new(0.8)

    local actual = criterion:forward(input, target)
    local expected = nn.BCECriterion():forward(input, target)
    tester:assertlt(actual, expected)
  end

  function suite.test_weighted_bce_gradient()
    local eps = 1e-2
    local input = torch.rand(10)*(1-eps) + eps/2
    local target = torch.rand(10)*(1-eps) + eps/2
    local criterion = WeightedBCECriterion.new(0.8)
    criterionJacobianTest(criterion, input, target)
  end

  tester:add(suite)

  return suite
end
