---
-- A binary cross-entropy criterion which allows you to weight the loss
-- differently based on the target. When `weight` is closer to 1, instances
-- where target = 1 will be weighted to have a higher loss than instances where
-- target = 0.
--
-- @classmod sconce.WeightedBCECriterion

require('nn')

local WeightedBCECriterion, Parent = require('sconce.ns').class('sconce.WeightedBCECriterion', 'nn.Criterion')

local eps = 1e-12

function WeightedBCECriterion:__init(weight, sizeAverage)
  Parent.__init(self)
  if weight ~= nil then
    self.weight = weight
  else
    self.weight = 0.5
  end
  if sizeAverage ~= nil then
    self.sizeAverage = sizeAverage
  else
    self.sizeAverage = true
  end
end


function WeightedBCECriterion:__len()
  return 0
end

function WeightedBCECriterion:updateOutput(input, target)
  -- -log(input) * target * weight - log(1 - input) * (1 - target)
  assert(input:nElement() == target:nElement(), 'input and target size mismatch')

  self.buffer = self.buffer or input.new()

  local buffer = self.buffer
  local weight = self.weight
  local output

  buffer:resizeAs(input)

  -- log(input) * target * weight * 2
  buffer:add(input, eps):log()

  if weight then
    buffer:mul(weight * 2)
  end

  output = torch.dot(target, buffer)

  -- log(1 - input) * (1 - target) * (1 - weight) * 2
  buffer:mul(input, -1):add(1):add(eps):log()

  if weight then
    buffer:mul((1 - weight) * 2)
  end

  output = output + torch.sum(buffer)
  output = output - torch.dot(target, buffer)

  if self.sizeAverage then
    output = output / input:nElement()
  end

  self.output = -output

  return self.output
end

function WeightedBCECriterion:updateGradInput(input, target)
  -- -(target * input * (1 - weight) - input + target * weight) / (input * (1 - input))
  -- The gradient is slightly incorrect:
  -- It should have be divided by (input + eps) (1 - input + eps)
  -- but it is divided by input (1 - input + eps) + eps
  -- This modification requires less memory to be computed.
  assert(input:nElement() == target:nElement(), 'input and target size mismatch')

  self.buffer = self.buffer or input.new()

  local buffer = self.buffer
  local gradInput = self.gradInput
  local weight = self.weight

  buffer:resizeAs(input)
  -- -input * (1 + eps -input) + eps
  buffer:add(input, -1):add(-eps):cmul(input):add(-eps)

  gradInput:resizeAs(input)
  if weight then
    -- (input * (target * (1 - 2 * weight) + weight - 1) + target * weight) * 2
    gradInput
      :mul(target, 1 - 2 * weight)
      :add(weight - 1)
      :cmul(input)
      :add(weight, target)
      :mul(2)
  else
    -- target - input
    gradInput:add(target, -1, input)
  end

  -- - (target - input) / (input * (1 + eps -input) + eps)
  gradInput:cdiv(buffer)

  if self.sizeAverage then
    gradInput:div(target:nElement())
  end

  return gradInput
end

return WeightedBCECriterion
