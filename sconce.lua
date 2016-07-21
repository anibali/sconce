----
-- @module sconce

local sconce = require('sconce.ns')

require('sconce.ClassificationMetrics')
require('sconce.DebugProbe')
require('sconce.ExpandRepeated')
require('sconce.ExtractFeaturesNN')
require('sconce.Freeze')
require('sconce.ProgressBar')
require('sconce.Sideload')
require('sconce.Swapout')
require('sconce.Trainer')
require('sconce.WeightedBCECriterion')

sconce.util = require('sconce.util')

return sconce
