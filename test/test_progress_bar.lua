local ProgressBar = require('sconce.ProgressBar')

return function(tester)
  local suite = torch.TestSuite()

  function suite.test_progress_bar()
    local pb = ProgressBar.new(1000)
    tester:eq(pb:is_finished(), false)
    tester:eq(pb:to_ascii(10), '| [----------]   0.00%')
    pb:advance(800)
    tester:eq(pb:is_finished(), false)
    tester:eq(pb:to_ascii(10), '/ [########--]  80.00%')
    pb:advance(200)
    tester:eq(pb:is_finished(), true)
    tester:eq(pb:to_ascii(10), '- [##########] 100.00%')
  end

  tester:add(suite)

  return suite
end
