	
??bU?q@
??bU?q@!
??bU?q@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-
??bU?q@???U@1?Bt?p@AΧ?UJϔ?IV?F??!@*	4^?I*g@2U
Iterator::Model::ParallelMapV2'???K??!?  ѥ?<@)'???K??1?  ѥ?<@:Preprocessing2F
Iterator::Model??&k?C??!?@?0??I@)?٬?\m??1??&?R?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?7?ܘ???!C??W??9@)!%̴???1^??/?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??~??@??!??e?&x0@)?<??+??1?i???B%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP?R)??!2t?k[@)P?R)??12t?k[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???B????!??K?H?@)???B????1??K?H?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?%jj٢?!?@????3@)F?2??y?1?0tZ?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??V??,??!??\?mH@)??.ow?1ݓ|?L@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??I3?@Q?a;ϼ?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???U@???U@!???U@      ??!       "	?Bt?p@?Bt?p@!?Bt?p@*      ??!       2	Χ?UJϔ?Χ?UJϔ?!Χ?UJϔ?:	V?F??!@V?F??!@!V?F??!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??I3?@y?a;ϼ?W@