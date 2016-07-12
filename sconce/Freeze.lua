--[[

Wraps another module, adding the ability to freeze and unfreeze parameter
updates on that module.

Use Freeze:freeze() to disable parameter updates on the wrapped module, and
Freeze:unfreeze() to enable them. By default the module will be frozen.

* module: The module to wrap

--]]

local Freeze, Parent = require('sconce.ns').class('sconce.Freeze', 'nn.Module')

function Freeze:__init(module)
  Parent.__init(self)

  self.module = module
  self.module:zeroGradParameters()
  self.frozen = true
end

function Freeze:freeze()
  self.frozen = true
end

function Freeze:unfreeze()
  self.frozen = false
end

function Freeze:updateOutput(input)
  self.output = self.module:updateOutput(input)
  return self.output
end

function Freeze:updateGradInput(input, gradOutput)
  self.gradInput = self.module:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Freeze:accGradParameters(...)
  if not self.frozen then
    return self.module:accGradParameters(...)
  end
end

local function delegate(class, get_delegation_target, methods)
  for i,method in ipairs(methods) do
    class[method] = function(self, ...)
      local other = get_delegation_target(self)
      return other[method](other, ...)
    end
  end
end

delegate(Freeze, function(self) return self.module end, {
  'clearState', 'evaluate', 'parameters', 'reset', 'training',
  'updateParameters', 'zeroGradParameters'
})

return Freeze
