--[[

Performs a forward pass through another network to derive features.

* extractor_nn: Feature extraction network
* extractor_func: Function which maps the state of extractor_nn after a forward
  pass to the output of this module (the default returns extractor_nn.output)

--]]

local ExtractFeaturesNN, parent = require('sconce.ns').class('sconce.ExtractFeaturesNN', 'nn.Module')

function ExtractFeaturesNN:__init(extractor_nn, extractor_func)
  parent.__init(self)

  self.extractor_nn = extractor_nn
  assert(torch.isTypeOf(self.extractor_nn, 'nn.Module'))

  self.extractor_func = extractor_func
  self.extractor_func = self.extractor_func or function(model) return model.output end
  assert(type(self.extractor_func) == 'function')
end

function ExtractFeaturesNN:updateOutput(input)
  self.extractor_nn:forward(input)
  self.output = self.extractor_func(self.extractor_nn)
  return self.output
end

function ExtractFeaturesNN:clearState()
  self.output = nil
  self.extractor_nn:clearState()
end
