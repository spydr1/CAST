
P
random_normal/shapeConst*
dtype0*%
valueB"              
?
random_normal/meanConst*
valueB
 *    *
dtype0
A
random_normal/stddevConst*
valueB
 *  �?*
dtype0
~
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
seed2 *

seed *
T0
[
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0
D
random_normalAddrandom_normal/mulrandom_normal/mean*
T0
`
Conv_30/weightsConst*9
value0B."���?h\e��l�>�����/? ��*
dtype0
^
Conv_30/weights/readIdentityConv_30/weights*
T0*"
_class
loc:@Conv_30/weights
C
Conv_30/biasesConst*
valueB"        *
dtype0
[
Conv_30/biases/readIdentityConv_30/biases*
T0*!
_class
loc:@Conv_30/biases
�
Conv_30/Conv2DConv2Drandom_normalConv_30/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_30/BiasAddBiasAddConv_30/Conv2DConv_30/biases/read*
T0*
data_formatNHWC
.
Conv_30/ReluReluConv_30/BiasAdd*
T0
X
Conv_31/weightsConst*1
value(B&"Xe��1�@��>Fh��*
dtype0
^
Conv_31/weights/readIdentityConv_31/weights*
T0*"
_class
loc:@Conv_31/weights
C
Conv_31/biasesConst*
valueB"        *
dtype0
[
Conv_31/biases/readIdentityConv_31/biases*
T0*!
_class
loc:@Conv_31/biases
�
Conv_31/Conv2DConv2DConv_30/ReluConv_31/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_31/BiasAddBiasAddConv_31/Conv2DConv_31/biases/read*
T0*
data_formatNHWC
.
Conv_31/ReluReluConv_31/BiasAdd*
T0
X
Conv_32/weightsConst*1
value(B&"���=O�}�?jt�(�+>*
dtype0
^
Conv_32/weights/readIdentityConv_32/weights*
T0*"
_class
loc:@Conv_32/weights
C
Conv_32/biasesConst*
valueB"        *
dtype0
[
Conv_32/biases/readIdentityConv_32/biases*
T0*!
_class
loc:@Conv_32/biases
�
Conv_32/Conv2DConv2DConv_31/ReluConv_32/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_32/BiasAddBiasAddConv_32/Conv2DConv_32/biases/read*
T0*
data_formatNHWC
.
Conv_32/ReluReluConv_32/BiasAdd*
T0
X
Conv_33/weightsConst*1
value(B&"�4h��o> ��%�*
dtype0
^
Conv_33/weights/readIdentityConv_33/weights*
T0*"
_class
loc:@Conv_33/weights
C
Conv_33/biasesConst*
valueB"        *
dtype0
[
Conv_33/biases/readIdentityConv_33/biases*
T0*!
_class
loc:@Conv_33/biases
�
Conv_33/Conv2DConv2DConv_32/ReluConv_33/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_33/BiasAddBiasAddConv_33/Conv2DConv_33/biases/read*
data_formatNHWC*
T0
.
Conv_33/ReluReluConv_33/BiasAdd*
T0
X
Conv_34/weightsConst*1
value(B&"~�K?o?�����?*
dtype0
^
Conv_34/weights/readIdentityConv_34/weights*
T0*"
_class
loc:@Conv_34/weights
C
Conv_34/biasesConst*
valueB"        *
dtype0
[
Conv_34/biases/readIdentityConv_34/biases*
T0*!
_class
loc:@Conv_34/biases
�
Conv_34/Conv2DConv2DConv_33/ReluConv_34/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_34/BiasAddBiasAddConv_34/Conv2DConv_34/biases/read*
T0*
data_formatNHWC
.
Conv_34/ReluReluConv_34/BiasAdd*
T0
X
Conv_35/weightsConst*1
value(B&"�"Q?
{|?�n�>����*
dtype0
^
Conv_35/weights/readIdentityConv_35/weights*
T0*"
_class
loc:@Conv_35/weights
C
Conv_35/biasesConst*
valueB"        *
dtype0
[
Conv_35/biases/readIdentityConv_35/biases*
T0*!
_class
loc:@Conv_35/biases
�
Conv_35/Conv2DConv2DConv_34/ReluConv_35/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_35/BiasAddBiasAddConv_35/Conv2DConv_35/biases/read*
T0*
data_formatNHWC
.
Conv_35/ReluReluConv_35/BiasAdd*
T0
X
Conv_36/weightsConst*1
value(B&"�{;�v�?%�j���?*
dtype0
^
Conv_36/weights/readIdentityConv_36/weights*
T0*"
_class
loc:@Conv_36/weights
C
Conv_36/biasesConst*
valueB"        *
dtype0
[
Conv_36/biases/readIdentityConv_36/biases*
T0*!
_class
loc:@Conv_36/biases
�
Conv_36/Conv2DConv2DConv_35/ReluConv_36/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_36/BiasAddBiasAddConv_36/Conv2DConv_36/biases/read*
data_formatNHWC*
T0
.
Conv_36/ReluReluConv_36/BiasAdd*
T0
X
Conv_37/weightsConst*1
value(B&"�$�?��� �=�����*
dtype0
^
Conv_37/weights/readIdentityConv_37/weights*
T0*"
_class
loc:@Conv_37/weights
C
Conv_37/biasesConst*
valueB"        *
dtype0
[
Conv_37/biases/readIdentityConv_37/biases*
T0*!
_class
loc:@Conv_37/biases
�
Conv_37/Conv2DConv2DConv_36/ReluConv_37/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_37/BiasAddBiasAddConv_37/Conv2DConv_37/biases/read*
T0*
data_formatNHWC
.
Conv_37/ReluReluConv_37/BiasAdd*
T0
X
Conv_38/weightsConst*1
value(B&" �>_ʆ?%>�P�>*
dtype0
^
Conv_38/weights/readIdentityConv_38/weights*"
_class
loc:@Conv_38/weights*
T0
C
Conv_38/biasesConst*
valueB"        *
dtype0
[
Conv_38/biases/readIdentityConv_38/biases*
T0*!
_class
loc:@Conv_38/biases
�
Conv_38/Conv2DConv2DConv_37/ReluConv_38/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_38/BiasAddBiasAddConv_38/Conv2DConv_38/biases/read*
data_formatNHWC*
T0
.
Conv_38/ReluReluConv_38/BiasAdd*
T0
X
Conv_39/weightsConst*1
value(B&"p�z���5�n�L?��.?*
dtype0
^
Conv_39/weights/readIdentityConv_39/weights*
T0*"
_class
loc:@Conv_39/weights
C
Conv_39/biasesConst*
valueB"        *
dtype0
[
Conv_39/biases/readIdentityConv_39/biases*
T0*!
_class
loc:@Conv_39/biases
�
Conv_39/Conv2DConv2DConv_38/ReluConv_39/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_39/BiasAddBiasAddConv_39/Conv2DConv_39/biases/read*
T0*
data_formatNHWC
.
Conv_39/ReluReluConv_39/BiasAdd*
T0
X
Conv_40/weightsConst*1
value(B&"@��ɀ��Q=b���*
dtype0
^
Conv_40/weights/readIdentityConv_40/weights*
T0*"
_class
loc:@Conv_40/weights
C
Conv_40/biasesConst*
valueB"        *
dtype0
[
Conv_40/biases/readIdentityConv_40/biases*!
_class
loc:@Conv_40/biases*
T0
�
Conv_40/Conv2DConv2DConv_39/ReluConv_40/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_40/BiasAddBiasAddConv_40/Conv2DConv_40/biases/read*
T0*
data_formatNHWC
.
Conv_40/ReluReluConv_40/BiasAdd*
T0
X
Conv_41/weightsConst*1
value(B&"��� �x�>�ʕ?*
dtype0
^
Conv_41/weights/readIdentityConv_41/weights*
T0*"
_class
loc:@Conv_41/weights
C
Conv_41/biasesConst*
valueB"        *
dtype0
[
Conv_41/biases/readIdentityConv_41/biases*
T0*!
_class
loc:@Conv_41/biases
�
Conv_41/Conv2DConv2DConv_40/ReluConv_41/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_41/BiasAddBiasAddConv_41/Conv2DConv_41/biases/read*
T0*
data_formatNHWC
.
Conv_41/ReluReluConv_41/BiasAdd*
T0
X
Conv_42/weightsConst*1
value(B&"~/`?|�>���u�?*
dtype0
^
Conv_42/weights/readIdentityConv_42/weights*
T0*"
_class
loc:@Conv_42/weights
C
Conv_42/biasesConst*
valueB"        *
dtype0
[
Conv_42/biases/readIdentityConv_42/biases*
T0*!
_class
loc:@Conv_42/biases
�
Conv_42/Conv2DConv2DConv_41/ReluConv_42/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_42/BiasAddBiasAddConv_42/Conv2DConv_42/biases/read*
T0*
data_formatNHWC
.
Conv_42/ReluReluConv_42/BiasAdd*
T0
X
Conv_43/weightsConst*1
value(B&"n�a?�9���*��%?*
dtype0
^
Conv_43/weights/readIdentityConv_43/weights*"
_class
loc:@Conv_43/weights*
T0
C
Conv_43/biasesConst*
valueB"        *
dtype0
[
Conv_43/biases/readIdentityConv_43/biases*
T0*!
_class
loc:@Conv_43/biases
�
Conv_43/Conv2DConv2DConv_42/ReluConv_43/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_43/BiasAddBiasAddConv_43/Conv2DConv_43/biases/read*
data_formatNHWC*
T0
.
Conv_43/ReluReluConv_43/BiasAdd*
T0
X
Conv_44/weightsConst*1
value(B&"�l��lC��.�y?p2X>*
dtype0
^
Conv_44/weights/readIdentityConv_44/weights*
T0*"
_class
loc:@Conv_44/weights
C
Conv_44/biasesConst*
valueB"        *
dtype0
[
Conv_44/biases/readIdentityConv_44/biases*
T0*!
_class
loc:@Conv_44/biases
�
Conv_44/Conv2DConv2DConv_43/ReluConv_44/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_44/BiasAddBiasAddConv_44/Conv2DConv_44/biases/read*
T0*
data_formatNHWC
.
Conv_44/ReluReluConv_44/BiasAdd*
T0
X
Conv_45/weightsConst*1
value(B&"�%�?�c�= Ė��`�=*
dtype0
^
Conv_45/weights/readIdentityConv_45/weights*
T0*"
_class
loc:@Conv_45/weights
C
Conv_45/biasesConst*
valueB"        *
dtype0
[
Conv_45/biases/readIdentityConv_45/biases*
T0*!
_class
loc:@Conv_45/biases
�
Conv_45/Conv2DConv2DConv_44/ReluConv_45/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

_
Conv_45/BiasAddBiasAddConv_45/Conv2DConv_45/biases/read*
T0*
data_formatNHWC
.
Conv_45/ReluReluConv_45/BiasAdd*
T0
X
Conv_46/weightsConst*
dtype0*1
value(B&"r�V?�����Z���?
^
Conv_46/weights/readIdentityConv_46/weights*
T0*"
_class
loc:@Conv_46/weights
C
Conv_46/biasesConst*
valueB"        *
dtype0
[
Conv_46/biases/readIdentityConv_46/biases*
T0*!
_class
loc:@Conv_46/biases
�
Conv_46/Conv2DConv2DConv_45/ReluConv_46/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
_
Conv_46/BiasAddBiasAddConv_46/Conv2DConv_46/biases/read*
data_formatNHWC*
T0
.
Conv_46/ReluReluConv_46/BiasAdd*
T0
X
Conv_47/weightsConst*
dtype0*1
value(B&"�z���q����@G>
^
Conv_47/weights/readIdentityConv_47/weights*
T0*"
_class
loc:@Conv_47/weights
C
Conv_47/biasesConst*
valueB"        *
dtype0
[
Conv_47/biases/readIdentityConv_47/biases*
T0*!
_class
loc:@Conv_47/biases
�
Conv_47/Conv2DConv2DConv_46/ReluConv_47/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
_
Conv_47/BiasAddBiasAddConv_47/Conv2DConv_47/biases/read*
T0*
data_formatNHWC
.
Conv_47/ReluReluConv_47/BiasAdd*
T0
X
Conv_48/weightsConst*1
value(B&"탋�ğ�>B�A?h ?>*
dtype0
^
Conv_48/weights/readIdentityConv_48/weights*"
_class
loc:@Conv_48/weights*
T0
C
Conv_48/biasesConst*
valueB"        *
dtype0
[
Conv_48/biases/readIdentityConv_48/biases*!
_class
loc:@Conv_48/biases*
T0
�
Conv_48/Conv2DConv2DConv_47/ReluConv_48/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_48/BiasAddBiasAddConv_48/Conv2DConv_48/biases/read*
data_formatNHWC*
T0
.
Conv_48/ReluReluConv_48/BiasAdd*
T0
X
Conv_49/weightsConst*1
value(B&"�󋿎�m?��V?��m�*
dtype0
^
Conv_49/weights/readIdentityConv_49/weights*
T0*"
_class
loc:@Conv_49/weights
C
Conv_49/biasesConst*
valueB"        *
dtype0
[
Conv_49/biases/readIdentityConv_49/biases*
T0*!
_class
loc:@Conv_49/biases
�
Conv_49/Conv2DConv2DConv_48/ReluConv_49/weights/read*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
_
Conv_49/BiasAddBiasAddConv_49/Conv2DConv_49/biases/read*
T0*
data_formatNHWC
.
Conv_49/ReluReluConv_49/BiasAdd*
T0 