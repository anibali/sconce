---
-- Adds Gaussian noise to the input tensor.
--
-- @classmod sconce.AddGaussianNoise

require('nn')

local AddGaussianNoise, Parent = require('sconce.ns').class('sconce.AddGaussianNoise', 'nn.Module')

function AddGaussianNoise:__init(stddev)
  Parent.__init(self)
  self.stddev = stddev
end

function AddGaussianNoise:updateOutput(input)
  self.output:resize(input:size())
  self.output:normal(0, self.stddev)
  self.output:add(input)
  return self.output
end

function AddGaussianNoise:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resizeAs(input):copy(gradOutput)

    return self.gradInput
  end
end

return AddGaussianNoise
