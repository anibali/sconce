---
-- Inserts a debug probe in the model.
--
-- Do not try to serialize a model with debug probes in it.
--
-- @classmod sconce.DebugProbe

local DebugProbe, parent = require('sconce.ns').class('sconce.DebugProbe', 'nn.Module')

function DebugProbe:__init(show_value, show_gradient)
  parent.__init(self)

  self.show_value = show_value or print
  self.show_gradient = show_gradient or function(...) end
  assert(type(self.show_value) == 'function')
  assert(type(self.show_gradient) == 'function')
end

function DebugProbe:updateOutput(input)
  self.show_value(input)

  self.output = input
  return self.output
end

function DebugProbe:updateGradInput(input, gradOutput)
  self.show_gradient(gradOutput)

  self.gradInput = gradOutput
  return self.gradInput
end
