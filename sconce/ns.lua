local torch = require('torch')

-- local ns_name = 'sconce'
local ns = {}

-- Function for creating a new class in the namespace
function ns.class(...)
  local args = {...}
  table.insert(args, ns)
  return torch.class(unpack(args))
end

return ns
