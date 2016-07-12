require('optim')
local Freeze = require('sconce.Freeze')

local function do_parameter_update(model, input, target, learning_rate)
  local criterion = nn.MSECriterion()

  local x, dloss_dx = model:getParameters()

  local feval = function(x_new)
    if x ~= x_new then
      x:copy(x_new)
    end

    dloss_dx:zero()
    local output = model:forward(input)
    local loss = criterion:forward(output, target)

    local dloss_dout = criterion:backward(output, target)
    model:backward(input, dloss_dout)

    return loss, dloss_dx
  end

  optim.sgd(feval, x, { learningRate = learning_rate }, {})
end

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_freeze_forward()
    local layer = nn.Mul()
    layer.weight:fill(1)
    local input = torch.Tensor{1, 2}
    local freeze = Freeze.new(layer)
    local actual = freeze:forward(input)
    local expected = layer:forward(input)
    tester:eq(actual, expected)
  end

  function suite.test_freeze_update()
    local input = torch.Tensor{1, 2}
    local target = input * 2

    local layer = nn.Mul()
    layer.weight:fill(1)

    local model = Freeze.new(layer)

    do_parameter_update(model, input, target, 0.1)
    tester:eq(layer.weight, torch.Tensor{1})
  end

  function suite.test_freeze_unfreeze()
    local input = torch.Tensor{1, 2}
    local target = input * 2

    local layer = nn.Mul()
    layer.weight:fill(1)

    local model = Freeze.new(layer)
    model:unfreeze()

    do_parameter_update(model, input, target, 0.1)
    tester:eq(layer.weight, torch.Tensor{1.5})
  end

  function suite.test_freeze_refreeze()
    local input = torch.Tensor{1, 2}
    local target = input * 2

    local layer = nn.Mul()
    layer.weight:fill(1)

    local model = Freeze.new(layer)
    model:unfreeze()

    do_parameter_update(model, input, target, 0.1)
    tester:eq(layer.weight, torch.Tensor{1.5})

    model:freeze()
    do_parameter_update(model, input, target, 0.1)
    tester:eq(layer.weight, torch.Tensor{1.5})
  end

  tester:add(suite)

  return suite
end
