require('nn')

local Swapout, Parent = require('sconce.ns').class('sconce.Swapout', 'nn.Module')

function Swapout:__init(thetas, inplace)
  Parent.__init(self)

  self.thetas = thetas
  self.inplace = inplace == true

  self.train = true
  self.masks = {}
end

function Swapout:updateOutput(input)
  if torch.type(self.thetas) == 'table' then
    assert(#self.thetas == #input, 'expected one theta per input')
  end

  if self.inplace then
    self.output:set(input[1])
  else
    self.output:resize(input[1]:size()):copy(input[1])
  end

  for i, input_i in ipairs(input) do
    local theta = torch.type(self.thetas) == 'number' and self.thetas or self.thetas[i]
    local mask = self.masks[i] or input_i.new()
    mask:resizeAs(input_i):bernoulli(theta)
    self.masks[i] = mask
    if i == 1 then
      self.output:cmul(mask)
    else
      self.output:addcmul(input_i, mask)
    end
  end
  return self.output
end

function Swapout:updateGradInput(input, gradOutput)
  self.gradInput = {}
  for i, input_i in ipairs(input) do
    local mask = self.masks[i]
    self.gradInput[i] = torch.cmul(gradOutput, mask)
  end
  return self.gradInput
end

function Swapout:clearState()
  for i, mask in ipairs(self.masks) do
    mask:set()
  end
  self.masks = {}
  return Parent.clearState(self)
end

return Swapout
