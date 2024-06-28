## Copyright (C) 2024 Pallav Purbia <pallavpurbia@gmail.com>
## Copyright (C) 2024 Andreas Bertsatos <abertsatos@biol.uoa.gr>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{Mdl} =} fitcnet (@var{X}, @var{Y})
## @deftypefnx {statistics} {@var{Mdl} =} fitcnet (@dots{}, @var{name}, @var{value})
##
## Fit a Neural Network classification model.
##
## @code{@var{Mdl} = fitcnet (@var{X}, @var{Y})} returns a Neural Network
## classification model, @var{Mdl}, with @var{X} being the predictor data,
## and @var{Y} the class labels of observations in @var{X}.
##
## @itemize
## @item
## @code{X} must be a @math{NxP} numeric matrix of input data where rows
## correspond to observations and columns correspond to features or variables.
## @var{X} will be used to train the neural network model.
## @item
## @code{Y} is @math{Nx1} matrix or cell matrix containing the class labels of
## corresponding predictor data in @var{X}. @var{Y} can contain numerical or categorical type
## of data. @var{Y} must have the same number of rows as @var{X}.
##
## @end itemize
##
## @code{@var{Mdl} = fitcnet (@dots{}, @var{name}, @var{value})} returns a
## Neural Network model with additional options specified by
## @qcode{Name-Value} pair arguments listed below.
##
## @multitable @columnfractions 0.05 0.4 0.75
## @headitem @tab @var{Name} @tab @var{Value}
##
## @item @tab @qcode{"NumLayers"} @tab Specifies the number of hidden layers.
## The default value is 1.
##
## @item @tab @qcode{"NumNeurons"} @tab A vector specifying the number of neurons in each layer.
## The default value is [10].
##
##
## @item @tab @qcode{"ActivationFunction"} @tab
## Specifies the activation function used
## in the neural network. It accepts the
## following options: (Default is 'ReLU')
##
## @itemize
##
## @item 'ReLU': Rectified Linear Unit (ReLU)
## is the default activation function. It
## outputs the input directly if it is
## positive, otherwise, it outputs zero.
## ReLU helps to solve the vanishing
## gradient problem and allows for faster
## training of deep networks.
##
## @item 'Sigmoid': The sigmoid function
## outputs a value between 0 and 1,
## representing a probability. It is
## commonly used in binary classification
## problems. However, it can suffer from
## vanishing gradients during
## backpropagation.
##
## @item 'Tanh': The hyperbolic tangent
## function outputs values between -1 and
## 1. It is zero-centered, making it
## preferable to the sigmoid function in
## some cases, but it also suffers from
## the vanishing gradient problem.
##
## @item 'Softmax': The softmax function is
## often used in the output layer of a neural
## network for multi-class classification
## problems. It converts the raw output scores
## into probabilities, where each classs
## probability is between 0 and 1, and the sum
## of all probabilities is 1.
##
## @end itemize
##
## @item @tab @qcode{"LearningRate"} @tab Specifies the learning rate for training the network.
## The default value is 0.01.
##
## @item @tab @qcode{"Epochs"} @tab Specifies the number of training epochs. The default value is 100.
##
## @item @tab @qcode{"BatchSize"} @tab Specifies the batch size for training. The default value is 32.
##
## @item @tab @qcode{"ValidationData"} @tab Specifies the validation data as a structure with fields 'XVal' and 'YVal'.
## Used for validation during training.
##
## @item @tab @qcode{"Verbose"} @tab Specifies whether to display training progress. It accepts either 0 or 1.
## The default value is 1.
##
## @end multitable
##
## @seealso{ClassificationNeuralNetwork, neuralnetworktrain, neuralnetworkpredict}
## @end deftypefn

function Mdl = fitcnet (X, Y, varargin)

  ## Check input parameters
  if (nargin < 2)
    error ("fitcnet: too few arguments.");
  endif
  if (mod (nargin, 2) != 0)
    error ("fitcnet: Name-Value arguments must be in pairs.");
  endif

  ## Check predictor data and labels have equal rows
  if (rows (X) != rows (Y))
    error ("fitcnet: number of rows in X and Y must be equal.");
  endif
  
  ## Check predictor data is numeric
  if ~isnumeric(X)
    error ("fitcnet: X must be a numeric matrix.");
  endif

  ## Check for missing values in X and Y
  if any(isnan(X(:))) || any(isnan(Y(:)))
    error ("fitcnet: X and Y must not contain missing values.");
  endif
  
  ## Parse arguments to class def function
  Mdl = ClassificationNeuralNetwork (X, Y, varargin{:});

endfunction


%!demo
## No demo for now.

## Test constructor
%!test
## No test for now.

## Test input validation
%!error<fitcnet: too few arguments.> fitcnet ()
%!error<fitcnet: too few arguments.> fitcnet (ones (4,1))
%!error<fitcnet: Name-Value arguments must be in pairs.>
%! fitcnet (ones (4,2), ones (4, 1), 'NumLayers')
%!error<fitcnet: number of rows in X and Y must be equal.>
%! fitcnet (ones (4,2), ones (3, 1))
%!error<fitcnet: number of rows in X and Y must be equal.>
%! fitcnet (ones (4,2), ones (3, 1), 'NumLayers', 2)
%!error<fitcnet: X must be a numeric matrix.>
%! fitcnet ({'a', 'b'; 'c', 'd'}, ones (2, 1))
%!error<fitcnet: X and Y must not contain missing values.>
%! fitcnet ([1, 2; NaN, 4], ones (2, 1))
%!error<fitcnet: X and Y must not contain missing values.>
%! fitcnet (ones (2, 2), [1; NaN])
