---
-- Creates a two-element table containing the input and a sideloaded tensor.
--
-- @classmod sconce.Sideload

require('nn')

local Sideload, Parent = require('sconce.ns').class('sconce.Sideload', 'nn.Module')

---
-- @tparam func get_tensor Function which returns the sideloaded tensor
function Sideload:__init(get_tensor)
  Parent.__init(self)
  assert(type(get_tensor) == 'function')
  self.get_tensor = get_tensor
end

function Sideload:updateOutput(input)
  local tensor = self.get_tensor()
  self.output = {input, tensor}
  return self.output
end

function Sideload:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput[1]
  return self.gradInput
end

return Sideload
