local Swapout = require('sconce.Swapout')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_swapout_always_add()
    local swapout = Swapout.new(1.0)
    local actual = swapout:forward({torch.Tensor{1, 2}, torch.Tensor{3, 2}})
    local expected = torch.Tensor{4, 4}
    tester:eq(actual, expected)
  end

  function suite.test_swapout_never_add()
    local swapout = Swapout.new(0.0)
    local actual = swapout:forward({torch.Tensor{1, 2}, torch.Tensor{3, 2}})
    local expected = torch.Tensor{0, 0}
    tester:eq(actual, expected)
  end

  function suite.test_swapout_sometimes_add()
    torch.manualSeed(1234)
    local swapout = Swapout.new(0.5)
    local actual = swapout:forward({
      torch.Tensor(4):fill(1), torch.Tensor(4):fill(2)
    })
    local expected = torch.Tensor{3, 1, 0, 0}
    tester:eq(actual, expected)
  end

  tester:add(suite)

  return suite
end
