local cjson = require('cjson')
local _ = require('downstroke')
local pl = require('pl.import_into')()
local itorch_util = require('itorch.util')
local base64 = require('base64')

local M = {}

function M.script_dir()
  local str = debug.getinfo(2, "S").source:sub(2)
  local dir = pl.path.abspath(str:match("(.*/)"))
  if _.ends_with(dir, '/.') then
    dir = dir:sub(1, #dir - 2)
  end
  return dir
end

function M.lerp(beta, start, finish)
  return start + beta * (finish - start)
end

-- Calculate mean with confidence interval using statistical bootstrapping
function M.bootstrap_mean(values, opts)
  local opts = opts or {}
  local iterations = opts.iterations or 10000
  local confidence = opts.confidence or 0.8

  local mean = values:mean()
  local resampled_means = torch.Tensor(iterations)

  for i=1,iterations do
    local resampled = torch.Tensor(values:size(1)):apply(function()
      return values[torch.random(1, values:size(1))]
    end)
    resampled_means[i] = resampled:mean()
  end
  resampled_means = resampled_means:sort()

  return {
    mean = resampled_means:mean(),
    confidence_lower = resampled_means[((1 - confidence) / 2) * iterations],
    confidence_upper = resampled_means[(1 - (1 - confidence) / 2) * iterations]
  }
end

function M.sanitise_file_name(unsanitised)
  return string.gsub(unsanitised, '[^a-zA-Z0-9_\\-\\.]', '_')
end

-- Reads and parses entire JSON file
function M.read_json_file(path)
  local file = io.open(path, 'r')
  if not file then return nil end
  local text = file:read('*all')
  file:close()
  return cjson.decode(text)
end

-- Writes entire JSON file
function M.write_json_file(path, value)
  local file = io.open(path, 'w')
  file:write(cjson.encode(value))
  file:close()
end

function M.graphviz_to_html(g)
  -- Export graph to a temporary location. This produces
  -- a .svg and a .dot file
  local filename = os.tmpname()
  graph.dot(g, 'Title', filename)

  -- Delete the .dot file
  os.remove(filename .. '.dot')

  -- Read and base 64 encode the model graph SVG image
  local file = io.open(filename .. '.svg', 'r')
  local image_b64 = base64.encode(file:read('*all'))
  file:close()

  -- Delete the .svg file
  os.remove(filename .. '.svg')

  return '<img style="width: 100%;" src=data:image/svg+xml;base64,' .. image_b64 .. '>'
end

function M.display_graphviz(g)
  itorch.html(M.graphviz_to_html(g))
end

-- Helper function for displaying nngraph model visualisations in an
-- iTorch notebook
function M.visualise_model(model)
  M.display_graphviz(model.fg)
end

function M.clear_itorch_output()
  local clear_msg = itorch_util.msg('clear_output', itorch._msg)
  clear_msg.content = {wait = false}
  itorch_util.ipyEncodeAndSend(itorch._iopub, clear_msg)
end

function M.one_hot(class, n_classes)
  return _.map(_.range(1, n_classes), function(i)
    return i == class and 1 or 0
  end)
end

function M.concat_lists(...)
  local joined = {}
  local lists = {...}
  for i=1,#lists do
    for j,value in ipairs(lists[i]) do
      table.insert(joined, value)
    end
  end
  return joined
end

function M.freeze_layer(layer)
  -- layer.old_accGradParameters = layer.accGradParameters
  layer.accGradParameters = function() end
end

-- function M.unfreeze_layer(layer)
--   layer.accGradParameters = layer.old_accGradParameters
-- end

function M.split_sequential(seq, sizes)
  local new_seqs = {}
  local offset = 0
  for i,size in ipairs(sizes) do
    local new_seq = nn.Sequential()
    for j = (offset + 1), (offset + size) do
      new_seq:add(seq:get(j))
    end
    offset = offset + size

    table.insert(new_seqs, new_seq)
  end
  return new_seqs
end

local function annotate(opts)
  local annotations = {
    name = opts.name,
    description = opts.description,
    graphAttributes = {
      color = opts.color,
      style = 'filled',
      fillcolor = opts.fillcolor
    }
  }

  return _.partial(nngraph.Node.annotate, _, annotations)
end

local layer_colours = {
  ['nn.SpatialBatchNormalization']  = 'darksalmon',
  ['nn.VolumetricConvolution']      = 'lightskyblue',
  ['nn.VolumetricMaxPooling']       = 'khaki',
  ['nn.VolumetricAveragePooling']   = 'greenyellow',
  ['nn.Linear']                     = 'gray',
  ['nn.ReLU']                       = 'plum',
  ['nn.ELU']                        = 'plum',
  ['nn.Dropout']                    = 'lightgreen'
}

function M.construct_model(...)
  local functions = {...}

  local coat_in_sugar = function(func)
    local colour = layer_colours[torch.type(func)]

    if colour then
      return _.flow(func, annotate{fillcolor = colour})
    end

    return func
  end

  return _.flow(unpack(_.map(functions, coat_in_sugar)))
end

return M
