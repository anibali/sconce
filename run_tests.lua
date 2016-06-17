require('torch')
local pl = require('pl.import_into')()

local tester = torch.Tester()

local function register_test_suite(name)
  require('test.' .. name)(tester)
end

for i,v in ipairs(pl.dir.getfiles('test', 'test_*.lua')) do
  register_test_suite(v:match('test/(.*).lua'))
end

tester:run()

