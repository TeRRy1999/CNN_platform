<!DOCTYPE html>
<html>

<head>
	<title>CNN - d3</title>
	<link rel="stylesheet" type="text/css" href="css/mystyle.css">
	<script src="js/jquery-3.3.1.min.js"></script>
	<script src="js/jquery-loading-master/dist/jquery.loading.js"></script>
	<link href="js/jquery-loading-master/dist/jquery.loading.css" rel="stylesheet">
	<script src="http://code.jquery.com/jquery-1.11.0.min.js"></script>
	<link rel="stylesheet" type="text/css" href="css/css.css">
	<script type="text/javascript" src="js/d3.v3.min.js"></script>
	<script type="text/javascript" src="js/javascript.js"></script>
	<link href="https://gitcdn.github.io/bootstrap-toggle/2.2.2/css/bootstrap-toggle.min.css" rel="stylesheet">
	<script src="https://gitcdn.github.io/bootstrap-toggle/2.2.2/js/bootstrap-toggle.min.js"></script>
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
  	<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
  	<script src="js/head.js"></script>
	<script src="js/sampleNN.js"></script>
</head>
<body>
<div style="background-color:black;width:130px;height:75px;position:absolute;left:47%;top:15%;z-index:1;display:none;" id="ycdt" onmouseout="(function(){$('#ycdt').css('display','none');})()"><img src="pictrue\\you_cannot_do_this.jpeg" width="130"></div>
<div class="form-row"> 
	<div class="col-md-12" style="height: 60px" id="Headerbar">
		<div class="form-row"> 
			<div class="col-md-3">
				<font color="blue" size="5" face="monospace" style="white-space:nowrap">ก็ʕ•͡ᴥ•ʔ ก้ &nbsp; dEEp Learning</font>
			</div>
			<div class="col-md-2">
			</div>
			<div class="col-md-2">
				<font color="blue" size="6" face="monospace" style="white-space:nowrap"><b>NN Module</b></font>
			</div>
			<div class="col-md-4">
			</div>
			<div class="col-md-1">
			</div>
		</div>
	</div>
</div>
<div>
	<ul class="nav nav-tabs" style="padding: 5px" id="tabs">
	    <li class="active"><a data-toggle="tab" href="#data">Data</a></li>
	    <li><a data-toggle="tab" href="#model">Model</a></li>
	    <li><a data-toggle="tab" href="#hyperParameters">HyperParameters</a></li>
	    <li><a data-toggle="tab" href="#training" onclick="show_result_page();check_can_run();">Training</a></li>
	    <li><a data-toggle="tab" href="#results">Results</a></li>
	    <li><a data-toggle="tab" href="#Inference">Inference/Deploy</a></li>
	</ul>
	<div class="tab-content">
	    <div id="data" class="tab-pane fade in active">
	    	<div class="form-row"> 
	    		<div class="col-md-2"></div>
				<div class="col-md-7">
					<div id="Data_setting">
						<div class="form-row" style="overflow: hidden;" >
							<div class="col-md-3">
								<font size="1" color="Gray">&nbsp;Dataset: </font>
								<select size="1" id="sample_project" class="form-control" onchange="chooseDataset();">
								<option value="no">Choose one...</option>
								<option value="MNIST">MNIST</option>
								<option value="CIFAR10">CFAR10</option>
								<option value="CATDOG">CATDOG</option>
								<option value="UPLOAD">Upload...</option>
								</select>
							</div>
							<div class="col-md-3">
								<font size="1" color="Gray">&nbsp;Train / Vaildation / Test </font>
								<select size="1" id="dataset_allocate" class="form-control" onchange="DatasetAllocate()">
								<option value="no">- - -</option>
								<option value="56000/7000/7000">80% / 10% / 10%</option>
								<option value="56000/3500/10500">80% / 5% / 15%</option>
								<option value="49000/10500/10500">70% / 15% / 15%</option>
								<option value="49000/14000/7000">70% / 20% / 10%</option>
								<option value="49000/7000/14000">70% / 10% / 20%</option>
								<option value="42000/7000/21000">60% / 10% / 30%</option>
								<option value="custom">Custom</option>
								</select>
							</div>
							<div class="col-md-6">
								<div id="datasetsNum_show_txt">
									<div class="col-md-4">
										<font size="1" color="Silver">Training</font>
										<span id="training_picNum_txt" class="form-control" style="height: 30px;" onmouseenter="(function(){$('#ycdt').css('display','');})()"></span>
									</div>
									<div class="col-md-4">
										<font size="1" color="Silver">Vaildation</font>
										<span id="vaildation_picNum_txt" class="form-control" style="height: 30px;"></span>
									</div>
									<div class="col-md-4">
										<font size="1" color="Silver">Test</font>
										<span id="test_picNum_txt" class="form-control" style="height: 30px;"></span>
									</div>
								</div>
								<div id="custom_input" style="display: none;">
									<div class="col-md-4">
										<font size="1" color="Silver">Training</font>
										<input id="training_picNum" type="text" class="form-control" style="height: 80%;">
									</div>
									<div class="col-md-4">
										<font size="1" color="Silver">Vaildation</font>
										<input id="vaildation_picNum" type="text" class="form-control" style="height: 80%;">
									</div>
									<div class="col-md-4">
										<font size="1" color="Silver">Test</font>
										<input id="test_picNum" type="text" class="form-control" style="height: 80%;">
									</div>
								</div>
								
							</div>
						</div>
						<div class="form-row" style="margin-top: 15px;margin-top: 20px;margin-left: 5px">
							<div class="col-md-3">
								<font size="1" color="Gray">&nbsp;Load Dataset in memory </font>
								<select size="1" id="sample_project" class="form-control" onchange="chooseDataset()">
								<option value="one_batch_at_a_time">One batch at a time</option>
								<option value="full_dataset">Full dataset</option>
								</select>
							</div>
							<div class="col-md-6">
								<label id="convAdvance_switch" onclick="shuffle_click();" class="switch" style="margin-top: 25px">
								  <input type="checkbox">
								  <span class="slider round"></span>
								  <font size="3" color="Gray" style="margin-left: 40px">shuffle</font>
								</label>
							</div>
							<div class="col-md-3">
								<div id="moduleImg_div" style="display: none" style="width:100%; margin-left: 10px;">
									<img id="moduleImg" style="width:80%;">
								</div>
								<div id="moduleUpload_div" style="display: none" style="width:100%; margin-left: 10px;">
									<font color="gray">Upload 3 zips like these: <br>train.zip <br>valid.zip <br>test.zip </font>
									<form id="upload_tainingData" method="post" enctype="multipart/form-data">
										<input type="file" id="upload_tainingData_file" name="file[]" multiple style="margin-top: 10px" accept="application/zip,application/rar" required/>
										<input type="submit" class="btn" value="Upload" style="box-shadow:0px 2px 1px 0px #aaaaaa;background-color: #00D1B6;margin-top: 10px"></input>
									</form>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col-md-1"></div>
				<div class="col-md-2" >
					<div id="Data_setting_comment">
						Click the select bar to choose a data set.
					</div>
					<div style="height: calc(100vh - 185px);border-left-style:solid;border-left-color:WhiteSmoke;">
					</div>
				</div>
			</div>
	    </div>
	    <div id="model" class="tab-pane fade">
	      <div class="form-row"> 
	      	<div class="col-md-10">
				<div class="form-row">
					<div id="Box3" class="col-md-2">
						<div>
							<font size="2">&nbsp;Sample model: </font>
							<select size="1" id="sampleNN" class="form-control" style="box-shadow:0px 2px 1px 0px #cccccc;" onchange="generateSample_NNmodule()">
								<option value="no">Choose one...</option>
								<option value="SAMPLE">sample</option>
								<option value="LeNet">LeNet</option>
								<option value="AlexNet">AlexNet</option>
								<option value="VGG">VGG</option>
								<option value="ResNet">ResNet</option>
							</select>
						</div>
						<br><br>
						<img class="img" id="Input" src="pictrue/input.png" alt="input" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Input');" onmouseout="ChangeBgColorBack('Input');">
						<img class="img" id="Output" src="pictrue/output.png" alt="output" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Output');" onmouseout="ChangeBgColorBack('Output');">
						<img class="img" id="Conv" src="pictrue/conv.png" alt="conv" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Conv');" onmouseout="ChangeBgColorBack('Conv');">
						<img class="img" id="Pool" src="pictrue/pool.png" alt="pool" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Pool');" onmouseout="ChangeBgColorBack('Pool');">
						<img class="img" id="Relu" src="pictrue/relu.png" alt="relu" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Relu');" onmouseout="ChangeBgColorBack('Relu');">
						<img class="img" id="Dropout" src="pictrue/dropout.png" alt="dropout" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Dropout');" onmouseout="ChangeBgColorBack('Dropout');">
						<img class="img" id="Linear" src="pictrue/linear.png" alt="linear" draggable="true" ondragstart="Drag(event)" onmouseover="ChangeBgColor('Linear');" onmouseout="ChangeBgColorBack('Linear');">
					</div>
					<div id="Box1" ondrop="Drop(event)" ondragover="AllowDrop(event)" class="col-md-10"></div>
				</div>
		  	</div>
		  	<div id="Status" class="col-md-2">
	  			<p id="the_intruduction"></p>
	  			<!-- <button id="savebtn" type="button" class="btn" style="position:absolute;right:5px; width: 30%; background-color: #F0F0F0; display: none" onclick="saveItemParameters()">save</button> -->
	  			<br><br>
	  			<!--///for Input -->
	  			<div id="div_inputParameters" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px;">input_channel*</span>
					  </div>
					  <input id="the_inputChannel" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">input size*</span>
					  </div>
					  <input id="the_inputSize" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<br>
				</div>
				<!--for Input ///-->

				<!--///for Output -->
				<!--for Output/// -->

	  			<!--///for Conv -->
	  			<div id="div_convParameters" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px;">kernal_size*</span>
					  </div>
					  <input id="the_kernalSize" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">filter num*</span>
					  </div>
					  <input id="the_filterNum" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<br>

					<label id="convAdvance_switch" onclick="conv_showAdvance_click();" class="switch">
					  <input type="checkbox">
					  <span class="slider round"></span>
					</label>
					<p>&nbsp;show advance<p> 
				</div>
				<div id="div_convAdvance" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">stride</span>
					  </div>
					  <input id="the_stride" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">padding</span>
					  </div>
					  <input id="the_padding" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">delation</span>
					  </div>
					  <input id="the_delation" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">groups</span>
					  </div>
					  <input id="the_groups" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
				</div>
				<!--for Conv/// -->

				<!--///for Pool -->
				<div id="div_poolParameters" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">pool size*</span>
					  </div>
					  <input id="the_poolSize" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
				</div>
				<!--for Pool/// -->

				<!--///for ReLu -->
				<!--for ReLu ///-->

				<!--///for Dropout -->
				<div id="div_dropoutParameters" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3" style="display: none">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">Drop*</span>
					  </div>
					  <input id="the_toDrop" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
				</div>
				<!--for Dropout/// -->

				<!--///for Linear -->
				<div id="div_linearParameters" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">out_Feature*</span>
					  </div>
					  <input id="the_outFeature" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
					<br>
					<label id="linearAdvance_switch" onclick="div_linearAdvance_click();" class="switch">
					  <input type="checkbox">
					  <span class="slider round"></span>
					</label>
					<p>&nbsp;show advance<p> 
				</div>
				<div id="div_linearAdvance" class="input-group mb-3" style="display: none">
					<div class="input-group mb-3">
					  <div class="input-group-prepend">
					    <span class="input-group-text" id="inputGroup-sizing-default" style="width: 100px">bias</span>
					  </div>
					  <input id="the_bias" type="text" class="form-control" aria-label="Default" aria-describedby="inputGroup-sizing-default" onchange="saveItemParameters();">
					</div>
				</div>
				<!--for Linear/// -->
		  	</div>
	      </div>
	    </div>
	    <div id="hyperParameters" class="tab-pane fade">
	      <div class="col-md-4"></div>
	      <div class="col-md-4" id="hpBar" style="margin-top: 20px">
      		<font color="Silver" class="col-md-12">epoch </font>
      		<div>
      			<div class="col-md-7">
      				<input id="epoch" type="text" class="form-control" value="3">
      			</div>
      			<div class="col-md-5"><p style="margin-top: 10px"></div>
      		</div>
      		<font color="Silver" class="col-md-12" style="margin-top: 10px">batch size </font>
      		<div>
      			<div class="col-md-7">
      				<input id="batch_size" type="text" class="form-control" value="50">
      			</div>
      			<div class="col-md-5"></div>
      		</div>
			<font color="Silver" class="col-md-12" style="margin-top: 10px">learning rate </font>
      		<div>
      			<div class="col-md-7">
      				<input id="lr" type="text" class="form-control" value="0.0001">
      			</div>
      			<div class="col-md-5"></div>
      		</div>
			<font color="Silver" class="col-md-12" style="margin-top: 10px">momentan </font>
      		<div>
      			<div class="col-md-7">
      				<input id="momentan" type="text" class="form-control" value="0.9">
      			</div>
      			<div class="col-md-5"></div>
      		</div>
			<!--<font color="Silver">weight decay </font>
			<select style="width:60%" size="1" id="weight_decay" class="form-control">
			<option value="0.001">0.001</option>
			<option value="0.01">0.01</option>
			<option value="0.1">0.1</option>
			<option value="1">1</option>
			<option value="10">10</option>
			</select>
			<br>-->
			<!--<font color="Silver">criterion </font>
			<select style="width:60%" size="1" id="criterion" class="form-control">
			<option value="CrossEntropyLoss">CrossEntropyLoss</option>
			<option value="Hinge">Hinge</option>
			<option value="Kullback">Kullback-Leibler</option>
			<option value="MAE">MAE</option>
			<option value="MSE">MSE</option>
			</select>
			<br>-->
			<div class="col-md-12" style="margin-top: 10px">
				<font color="Silver">optimizer </font>
				<select style="width:56%" size="1" id="optimizer" class="form-control">
				<option value="ADAM">Adam</option>
				<option value="SGD">SGD</option>
				</select>
			</div>
			<!--<option value="zero_grad">zero grad</option>
			<option value="SparseAdam">SparseAdam</option>
			<option value="Adamax">Adamax</option>
			<option value="ASGD">ASGD</option>
			<option value="LBFGS">LBFGS</option>-->
	      </div>
	    </div>
	    <div id="training" class="tab-pane fade">
	    	<div style="padding: 20px">
	    		<div class="col-md-12">
	    			<input type="button" style="position:absolute;right:250px;width: 10%;box-shadow:5px 6px 5px 0px #cccccc;background-color: #00D1B6;" value="Train" id="btn" class="btn"/>
	    			<button type="button" disabled style="position:absolute;right:5px;width: 10%;box-shadow:0px 2px 1px 0px #cccccc;background-color: #FFBF3F;" id="stopbtn" class="btn" onclick="stopclk();">reset</button>
			    </div>
			    <div class="form-row" style="margin-top: 50px">
			    	<div class="col-md-6">
						<div id="iframe">
						</div>
					</div>
					<div class="col-md-6">
						<div id="iframe_loss">
						</div>
					</div>
					<!-- <div class="col-md-12">
						<div id="iframe_progress">
						</div>
					</div> -->
			    </div>
	    	</div>
	    </div>
	    <div id="results" class="tab-pane fade">
	  		<ul class="nav nav-tabs" style="padding: 5px" id="tabs">
			    <li class="active"><a data-toggle="tab" href="#graghs">Graghs</a></li>
			    <li><a data-toggle="tab" href="#configuration">Configuration</a></li>
			    <li><a data-toggle="tab" href="#detail" onclick="result_detail_clk();">Detail</a></li>
			</ul>
			<div class="tab-content">
				<div id="graghs" class="tab-pane fade in active">
					<div style="padding: 20px">
			        	<font size="6">Result Graphs...</font>
			  		</div>
				</div>
				<div id="configuration" class="tab-pane fade">
			        <div id="config" style="padding: 20px; white-space:pre-wrap">
			        	<font size="6">Configuration...</font>
			  		</div>
				</div>
				<div id="detail" class="tab-pane fade">
					<div style="padding: 20px">
			        	<div id="detail_images" style="pading: 20px;min-height: 200px;">
			        		<div style="font-family:sans-serif;font-size:25px;color:gray;">weight & bias of every layer:</div>
			        	</div>
			        	<div id="detail_featureMaps" style="pading: 20px;margin-top: 50px;overflow: hidden;">
			        		<div style="font-family:sans-serif;font-size:25px;color:gray;">feature maps of every layer:</div>
			        		<div id="featureMaps_div" style="margin-left: 20px;margin-bottom: 20px;">
			        		</div>
			        	</div>
			  		</div>
				</div>
			</div>
	    </div>
	    <div id="Inference" class="tab-pane fade">
	    	<ul class="nav nav-tabs" style="padding: 5px" id="tabs">
			    <li class="active"><a data-toggle="tab" href="#dataset_inference">Dataset Inference</a></li>
			    <li><a data-toggle="tab" href="#form_inference">Form Inference</a></li>
			    <li><a data-toggle="tab" href="#downlaod">DownLoad</a></li>
			</ul>
			<div class="tab-content">
				<div id="dataset_inference" class="tab-pane fade in active">
					<div class="container" id="dataset_inference_selector">
						<div class="form-group">
							<font color="Silver">choose Dataset</font>
					    	<select size="1" id="chooseDataset" class="form-control" onchange="chooseDataset_change();">
								<option value="test">Test</option>
								<option value="upload">Upload</option>
							</select>
							<div id="upload_dataset_div" style="display: none;margin-top: 20px;border: 1px solid Gainsboro;padding: 10px;">
								<form action="/somewhere/to/upload" enctype="multipart/form-data">
									<input type="file" style="margin-top: 10px" onchange="readURL(this)" targetID="the_upload_dataset" accept=".zip,.rar,.7z"/>
									<input type="submit" class="btn" value="Upload" name="submit" style="width: 30%;box-shadow:0px 2px 1px 0px #cccccc;margin-top: 10px">
								</form>
							</div>
						    <button id="dataset_start_inference" type="button" class="btn" style="background-color: #00D1B6; margin-top: 40px;" onclick="dataset_start_inference()">Start Inference</button>
					  	</div>
					</div>
					<div id="dataset_inference_show_result" style="display: none;">
						<div class="col-md-3" style="height: 900px;overflow: hidden;"></div>
					    <div class="col-md-6" id="form_inference_result_content">
							<div style="padding: 20px;overflow: hidden;">
								<label id="show_testError_btn" onclick="show_testError_clk();" class="switch">
								  <input type="checkbox">
								  <span class="slider round"></span>
								  <font size="3" color="Gray" style="margin-left: 40px;">show_error_images</font> 
								</label>
								<div style="box-shadow:0px 1px 0px 0px #cccccc;overflow: hidden;">
									<div class="col-md-4"><font color="Silver">Picture:</font></div>
									<div class="col-md-4"><font color="Silver">Expect:</font></div>
									<div class="col-md-4"><font color="Silver">Predict:</font></div>
								</div>
								<div id="dataset_inference_show_result_content" style="height: 680px;overflow: auto;">
								</div>
								<button type="button" class="btn" style="margin-top: 40px;" onclick="dataset_inference_back()">Back</button>
							</div>
					    </div>
					</div>
				</div>
				<div id="form_inference" class="tab-pane fade">
					<div class="col-md-1"></div>
			        <div class="col-md-2">
			        	<button id="upload_my_py_pkl_btn" class="btn" style="box-shadow:0px 2px 1px 0px #aaaaaa;background-color: #00D1B6;margin-top: 10px;" onclick="upload_my_py_pkl_clk()">Upload my py and pkl</button>
			        	<div id="test_py_uploader" style="display: none;">
			        		<font color="Silver">Upload your py and pkl or not </font>
				        	<div id="upload_py_pkl_div" style="width:100%;">
								<font color="gray">like below: <br>test.py <br>model.pkl</font>
								<form id="upload_py_pkl_form" method="post" enctype="multipart/form-data">
									<input type="file" id="upload_py_pkl" name="file[]" multiple style="margin-top: 10px" accept="application/py,application/pkl" required/>
									<input type="submit" class="btn" value="Upload" style="box-shadow:0px 2px 1px 0px #aaaaaa;background-color: #00D1B6;margin-top: 10px"></input>
								</form>
							</div>
						</div>
			        </div>
				    <div class="col-md-3" id="form_inference_selector">
			    		<font color="Silver" size="4">Please upload a Picture for testing </font>
						<form id="uploadimage" method="post" enctype="multipart/form-data">
							<input type="file" id = "file" name="file" multiple style="margin-top: 10px" targetID="form_upload_img" accept="image/gif, image/jpeg, image/png" required/>
								<div id= "image_preview" style="width: 100%;height: 400px;border: 1px solid WhiteSmoke;">
									<img style="min-width: 60%;max-width: 100%;max-height:100%;width: auto; height: auto;" id="previewing" src="#"/>
								</div>
							<input type="submit" class="btn" value="Start Inference" style="position:absolute;right:20px;bottom: 20px;box-shadow:0px 2px 1px 0px #aaaaaa;background-color: #00D1B6;margin-top: 10px"></input>
						</form>
				    </div>
				    <div class="col-md-3" id="form_inference_result">
				    	<font color="Silver" size="4">result: </font>
				    	<h4 id='loading' >loading..</h4>
				    	<div id="form_result_div" style="width: 100%;height: 440px;border: 1px solid WhiteSmoke;margin-top: 35px">
				    		<div id="message"></div>
						</div>
				    </div>
				</div>
				<div id="downlaod" class="tab-pane fade">
			        <div class="container" id="upLoadBar" style="width: 400px;">
			        	<form>
						  	<div class="form-group" style="margin-top: 20px">
							    <a class="btn" style="width:60%;background-color: #00D1B6; margin-top: 20px;margin-left: 60px" href="php\download.php">Download Trained Model</a>
							    <a class="btn" style="width:60%;background-color: #FFBF3F; margin-top: 20px;margin-left: 60px" href="php\download_py.php">Download test.py</a>
						  	</div>
						</form>
			        </div>
				</div>
			</div>
	    </div>
	</div>
</div>
<input id="code" value="" style="display: none">
</div>
</body>
</html>

