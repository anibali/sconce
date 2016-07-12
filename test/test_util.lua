local util = require('sconce.util')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_lerp()
    local actual = util.lerp(0.5, 1, 2)
    local expected = 1.5
    tester:eq(actual, expected)
  end

  function suite.test_sanitise_file_name()
    local actual = util.sanitise_file_name('dir/Micro$oft.exe')
    local expected = 'dir_Micro_oft.exe'
    tester:eq(actual, expected)
  end

  function suite.test_one_hot()
    local actual = util.one_hot(2, 5)
    local expected = {0, 1, 0, 0, 0}
    tester:eq(actual, expected)
  end

  function suite.test_concat_lists()
    local actual = util.concat_lists({1, 2}, {3, 4, 5})
    local expected = {1, 2, 3, 4, 5}
    tester:eq(actual, expected)
  end

  function suite.test_split_sequential()
    local seq = nn.Sequential()
      :add(nn.Identity())
      :add(nn.Identity())
      :add(nn.Identity())
      :add(nn.Identity())
    local seq1, seq2 = unpack(util.split_sequential(seq, {3, 1}))
    tester:eq(seq1:size(), 3)
    tester:eq(seq2:size(), 1)
  end

  tester:add(suite)

  return suite
end
