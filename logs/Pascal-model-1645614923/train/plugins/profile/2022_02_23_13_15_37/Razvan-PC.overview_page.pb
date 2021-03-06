?	?@?M?r@?@?M?r@!?@?M?r@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?@?M?r@??<??+@1yY?q@AV]???I??£?)@*	??C??c@2U
Iterator::Model::ParallelMapV2EF$aߦ?!?ׄR>4<@)EF$aߦ?1?ׄR>4<@:Preprocessing2F
Iterator::Model?c?1??!!??xH]K@)??Z????1??R?R?:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???;ޤ?!????{?9@)r?߅?٢?1?O/Ӆ>7@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)?7Ӆ??!????(=@))?7Ӆ??1????(=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??tx???!3???kt-@)? ??=@??1?]???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor ?!p$p?!G+[??@) ?!p$p?1G+[??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapC?_?+כ?!?
u?=*1@)??Z??o?1??ً@?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipMN?S[??!?B???F@)!?J?n?1?0????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIH???!@Q??	p+?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??<??+@??<??+@!??<??+@      ??!       "	yY?q@yY?q@!yY?q@*      ??!       2	V]???V]???!V]???:	??£?)@??£?)@!??£?)@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qH???!@y??	p+?V@?"M
&Adam/Adam/update_212/ResourceApplyAdamResourceApplyAdamE?????!E?????"/
model/fc_1/MatMulMatMul?11x"???!?W?\????0"=
gradient_tape/model/fc_1/MatMulMatMul?m?(?;??!?9u??b??0"=
!gradient_tape/model/fc_1/MatMul_1MatMulc4?؅??!??????"o
Cgradient_tape/model/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter7????!S??+???0"o
Cgradient_tape/model/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterRL?ؽ???!???|?3??0"o
Cgradient_tape/model/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?H꾅?!??͈???0"f
:gradient_tape/model/conv1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter*?G???!+.?6?`??0"o
Cgradient_tape/model/conv5_block1_0_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?\??N?~?!???r????0"m
Bgradient_tape/model/conv5_block1_0_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?7^Ѫ{?!}???I??0Q      Y@Y+??>???at????X@q?Q^qt@y?֪XB]?"?

both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 