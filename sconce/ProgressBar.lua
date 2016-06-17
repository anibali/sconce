local ProgressBar = require('sconce.ns').class('sconce.ProgressBar')

local spinner = {'|', '/', '-', '\\'}

function ProgressBar:__init(max_value)
  self.current_value = 0
  self.max_value = max_value or 1
  self.spinner_index = 1

  return self
end

function ProgressBar:set_current_value(value)
  self.current_value = value

  self.spinner_index = self.spinner_index + 1
  if self.spinner_index > #spinner then self.spinner_index = 1 end
end

function ProgressBar:advance(amount)
  self:set_current_value(self.current_value + amount)
end

function ProgressBar:is_finished()
  return self.current_value >= self.max_value
end

function ProgressBar:to_ascii(bar_width)
  bar_width = bar_width or 20

  local bar_table = {}

  local progress = self.current_value / self.max_value
  if progress > 1 then progress = 1 end

  for k=1,bar_width do
    if k / bar_width <= progress then
      table.insert(bar_table, '#')
    else
      table.insert(bar_table, '-')
    end
  end

  local bar = table.concat(bar_table)

  local ascii = string.format('%s [%s] %6.2f%%',
    spinner[self.spinner_index], bar, 100 * progress)

  return ascii
end

return ProgressBar
