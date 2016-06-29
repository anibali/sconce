local _ = require('downstroke')
local torch = require('torch')
local pl = require('pl.import_into')()

local Trainer = require('sconce.ns').class('sconce.Trainer')

-- "Adding gradient noise improves learning for very deep networks"
-- - Neelakantan et al.
local function gradient_noise(grad, t, eta, gamma)
  local eta = eta or 0.01
  local gamma = gamma or 0.55

  local std_dev = torch.pow(t + 1, -gamma / 2) * torch.sqrt(eta)
  grad:add(std_dev, torch.randn(grad:size()):typeAs(grad))

  return grad
end

function Trainer:__init(model, criterion, optimiser)
  assert(model)
  self.model = model
  self.params, self.dloss_dparams = model:getParameters()

  assert(criterion)
  self.criterion = criterion

  assert(optimiser)
  self.optimiser = optimiser

  self.listeners = {}
  self.progress = 0
  self.transform_gradient = function(grad, t)
    return gradient_noise(grad:clamp(-10, 10), t)
  end
end

function Trainer:on(event_name, listener)
  self.listeners[event_name] = self.listeners[event_name] or {}
  table.insert(self.listeners[event_name], listener)
end

function Trainer:clear_listeners()
  self.listeners = {}
end

function Trainer:fire(event_name, event)
  event.trainer = self
  if self.listeners[event_name] then
    for i,listener in ipairs(self.listeners[event_name]) do
      listener(event)
    end
  end
end

-- Calculates the vector gradient of loss with respect to model parameters
function Trainer:calculate_loss_and_gradient(batch_inputs, batch_targets)
  -- 1. Compute actual batch outputs using our model
  local batch_outputs = self.model:forward(batch_inputs)
  -- 2. Compute the loss of these outputs (relative to target outputs)
  local loss = self.criterion:forward(batch_outputs, batch_targets)
  -- 3. Compute the derivative of the loss with respect to the model outputs
  local dloss_dout = self.criterion:backward(batch_outputs, batch_targets)
  -- 4. Using `dloss_dout` as a starting point, perform backpropagation to
  -- update `dloss_dparams`. To emphasise, doing a backwards pass on the model
  -- **will mutate** `dloss_dparams`.
  self.dloss_dparams:zero()
  self.model:backward(batch_inputs, dloss_dout)

  if self.transform_gradient then
    local transformed_dloss_dout = self.transform_gradient(dloss_dout, self.n_processed_examples)
    if transformed_dloss_dout ~= dloss_dout then
      dloss_dout:copy(transformed_dloss_dout)
    end
  end

  self:fire('loss-calculated', {
    batch_outputs = batch_outputs,
    batch_targets = batch_targets,
    loss = loss
  })

  -- Return the loss and gradient of loss with respect to model parameters
  return loss, self.dloss_dparams
end

function Trainer:do_optimisation_step(batch_inputs, batch_targets)
  self.model:training()

  -- Perform a single optimisation step to update the model parameters based
  -- on the loss gradient
  local new_params, loss = self.optimiser.method(
    _.partial(self.calculate_loss_and_gradient, self, batch_inputs, batch_targets),
    self.params,
    self.optimiser.state)

  -- Ensure that the model has its parameters updated
  if new_params ~= self.params then
    -- `parameters` is updated in place by optim, so this copy should never
    -- actually occur.
    self.params:copy(new_params)
  end

  -- Return the current training loss
  return loss[1]
end

function Trainer:copy_batch(dest_inputs, dest_targets, src_inputs, src_targets)
  local event = {
    original_batch_inputs = src_inputs,
    original_batch_targets = src_targets
  }
  self:fire('transform-batch', event)
  src_inputs = event.batch_inputs or src_inputs
  src_targets = event.batch_targets or src_targets

  dest_inputs:resize(src_inputs:size()):copy(src_inputs)
  dest_targets:resize(src_targets:size()):copy(src_targets)

  self:fire('batch-loaded', {
    batch_inputs = dest_inputs,
    batch_targets = dest_targets
  })

  return dest_inputs, dest_targets
end

function Trainer:validation_cost()
  return 0
end

function Trainer:train(data_loader, opts)
  assert(torch.isTypeOf(data_loader, 'dl.DataLoader'))
  opts = opts or {}
  assert(type(opts) == 'table')

  self.should_stop_training = nil
  self.progress = 0
  self.n_processed_examples = 0

  local sample_batches = opts.sample_batches == true
  local batch_size = opts.batch_size or 16
  local n_examples_per_epoch = opts.n_examples_per_epoch or data_loader:size()
  local n_epochs = opts.n_epochs or 10
  local smoothing_alpha = opts.smoothing_alpha or 1

  local batch_inputs = torch.Tensor(1):typeAs(self.params)
  local batch_targets = torch.Tensor(1):typeAs(self.params)

  local avg_params = self.params:float()
  local best_val_cost = self:validation_cost()
  local best_params = avg_params:clone()

  for epoch = 1, n_epochs do
    local epoch_loss = 0
    local n_iterations = 0

    data_loader:reset()
    local batch_iterator = nil
    if sample_batches then
      batch_iterator = data_loader:sampleiter(batch_size, n_examples_per_epoch)
    else
      batch_iterator = data_loader:subiter(batch_size, n_examples_per_epoch)
    end

    for n_loaded_this_epoch, inputs, targets in batch_iterator do
      local new_inputs, new_targets =
        self:copy_batch(batch_inputs, batch_targets, inputs, targets)

      local loss = self:do_optimisation_step(new_inputs, new_targets)

      self.n_processed_examples = self.n_processed_examples + inputs:size(1)
      epoch_loss = epoch_loss + loss

      n_iterations = n_iterations + 1
      self.progress = (epoch - 1 + n_loaded_this_epoch / n_examples_per_epoch) / n_epochs

      if self.should_stop_training then
        break
      end
    end

    epoch_loss = epoch_loss / n_iterations

    avg_params:mul(1 - smoothing_alpha)
    avg_params:add(smoothing_alpha, self.params:float())

    local val_cost = self:validation_cost()
    if val_cost <= best_val_cost then
      self.best_epoch = epoch
      best_val_cost = val_cost
      best_params:copy(avg_params)
    end

    self:fire('epoch-finished', {
      loss = epoch_loss,
      validation_cost = val_cost,
      epoch = epoch
    })

    if self.should_stop_training then
      break
    end
  end

  self.progress = 1

  self.params:copy(best_params)
end

function Trainer:stop_training()
  self.should_stop_training = true
end

return Trainer
