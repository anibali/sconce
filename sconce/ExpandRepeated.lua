require('nn')

local ExpandRepeated, Parent = require('sconce.ns').class('sconce.ExpandRepeated', 'nn.Module')

function ExpandRepeated:__init(n, dim, ndims)
  Parent.__init(self)
  self.n = n
  self.dim = dim
  self.ndims = ndims
end

function ExpandRepeated:updateOutput(input)
  local dim = self.dim + (input:dim() - self.ndims)
  assert(self.n >= input:size(dim), 'ExpandRepeated cannot shrink tensors')
  local size = input:size()
  size[dim] = self.n
  self.output:resize(size)
  for i = 1, self.n, input:size(dim) do
    local amount = math.min((self.n - i) + 1, input:size(dim))
    self.output:narrow(dim, i, amount):copy(input:narrow(dim, 1, amount))
  end
  return self.output
end

function ExpandRepeated:updateGradInput(input, gradOutput)
  if self.gradInput then
    local dim = self.dim + (input:dim() - self.ndims)
    self.gradInput:resizeAs(input):zero()
    for i = 1, self.n, input:size(dim) do
      local amount = math.min((self.n - i) + 1, input:size(dim))
      self.gradInput:narrow(dim, 1, amount):add(gradOutput:narrow(dim, i, amount))
    end

    return self.gradInput
  end
end

return ExpandRepeated
