require('nn')

local Sideload, Parent = require('sconce.ns').class('sconce.Sideload', 'nn.Module')

function Sideload:__init(get_tensor)
  Parent.__init(self)
  assert(type(get_tensor) == 'function')
  self.get_tensor = get_tensor
end

function Sideload:updateOutput(input)
  local tensor = self.get_tensor()
  return {input, tensor}
end

function Sideload:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput[1]
  return self.gradInput
end

return Sideload
