//add in 0816 with d3 for mv <<<<<
var ItemWidth = 200;
var ItemHeight = 50;
var width;
var height;
var svg;
var holding_line=0;
var _lineFrom="null";
var _lineTo="null";
var path_count=0;
var the_clicked_item="null";
var click_path = 0;
var conv_showAdvance = 0;
var linear_showAdvance = 0;
var count_show_resultError_clk = 0;
var shuffle_count = 0;
var shuffle = "false";
var CANNOT_CLICK = 0;
var CAN_RUN = 0;

var batch_size_frame = 1;
var epoch_frame = 0;
var training_picNum_frame = 0;

var do_canvas = 1;
//add in 0816 with d3 for mv >>>>>

function main(){
  default_file();
  deleteAllCookies();
  var val = document.getElementById("btn");
  val.addEventListener("click", check_to_run, false);

  

  upload_listener();
  upload_model_py_listener();
  upload_tainingData_listener();

  count = new Array(7);
  for(i=0;i<7;i++){
    count[i] = 0;
  }
  var code = document.getElementById("code");
  show_result_page();
  initialSVG();
}

function default_file(){


  $.ajax({

      url: "php/deletefile.php",

      type: "POST",
      datatype: "html",
      success: function(output) 
      {
          //alert("default_file");
          //$("#show").html(output);
      },
      error : function()
      {
         alert( "Request failed.\n" );
      }
  
});


}

function upload_listener(){
  $("#uploadimage").on('submit',(function(e) {
    e.preventDefault();
    $("#message").empty();
    $('#loading').show();
    $.ajax({
      url: "php/upload.php", // Url to which the request is send
      type: "POST",             // Type of request to be send, called as method
      data: new FormData(this), // Data sent to server, a set of key/value pairs (i.e. form fields and values)
      contentType: false,       // The content type used when sending data to the server.
      cache: false,             // To unable request pages to be cached
      processData:false,        // To send DOMDocument or non processed data file it is set to false
      success: function(data)   // A function to be called if request succeeds
      {
        alert(data);
        $('#loading').hide();
        $("#message").html(data);

        //show_test_result
        $("#form_result_div").empty();
        $.get('python/Test_Result.txt')
        .done(function (data) {
          $("#form_result_div").append(
            '<div style="font-family:sans-serif;font-size:25px;color:gray;">The picture is : '+data+'</div>'
            );
        })
        .fail(function (jqXHR, textStatus, errorThrown){ 
          var err = textStatus + ", " + errorThrown; 
          console.log( "Request Failed: " + err ); 
          alert("[error] Please train first.");
        });
      }
    });
  }));

  // Function to preview image after validation
  $(function() {
    $("#file").change(function() {
    $("#message").empty(); // To remove the previous error message
    
    var file = this.files[0];
    var imagefile = file.type;
    var match= ["image/jpeg","image/png","image/jpg"];
    if(!((imagefile==match[0]) || (imagefile==match[1]) || (imagefile==match[2])))
    {
      $('#previewing').attr('src','noimage.png');
      $("#message").html("<p id='error'>Please Select A valid Image File</p>"+"<h4>Note</h4>"+"<span id='error_message'>Only jpeg, jpg and png Images type allowed</span>");
      return false;
    }
    else
    {
      var reader = new FileReader();
      reader.onload = imageIsLoaded;
      reader.readAsDataURL(this.files[0]);
    }
    });
  });
  function imageIsLoaded(e) {
  $("#file").css("color","green");
  $('#image_preview').css("display", "block");
  $('#previewing').attr('src', e.target.result);
  $('#previewing').attr('width', '100%');
  $('#previewing').attr('height', '100%');
  };
}

function upload_tainingData_listener(){
  $("#upload_tainingData").on('submit',(function(e) {
    e.preventDefault();

    $.ajax({
      url: "php/upload_train_data.php", // Url to whichIFIFIF the request is sendIFIFIF
      type: "POST",             // Type of request to be send, called as method
      data: new FormData(this), // Data sent to server, a set of key/value pairs (i.e. form fields and values)
      contentType: false,       // The content type used when sending data to the server.
      cache: false,             // To unable request pages to be cached
      processData:false,        // To send DOMDocument or non processed data file it is set to false
      success: function(data)   // A function to be called if request succeeds
      {
        alert(data);
        $('#upload_tainingData_loading').hide();
      
      }
    });
  }));
}

function upload_model_py_listener(){
  $("#upload_py_pkl_form").on('submit',(function(e) {
    e.preventDefault();

    $.ajax({
      url: "php/upload_model_py.php", // Url to whichIFIFIF the request is sendIFIFIF
      type: "POST",             // Type of request to be send, called as method
      data: new FormData(this), // Data sent to server, a set of key/value pairs (i.e. form fields and values)
      contentType: false,       // The content type used when sending data to the server.
      cache: false,             // To unable request pages to be cached
      processData:false,        // To send DOMDocument or non processed data file it is set to false
      success: function(data)   // A function to be called if request succeeds
      {
        alert(data);
      
      }
    });
  }));
}

function deleteAllCookies() {
    var cookies = document.cookie.split(";");

    for (var i = 0; i < cookies.length; i++) {
        var cookie = cookies[i];
        var eqPos = cookie.indexOf("=");
        var name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
    }
}


function initialSVG(){

  for(i=0;i<7;i++){
    count[i] = 0;
  }
  conv_showAdvance = 0;
  linear_showAdvance = 0;
  click_path = 0;
  the_clicked_item="null";
  path_count=0;
  holding_line=0;
  _lineTo="null";
  _lineFrom="null";
  //add in 0816 with d3 for mv <<<<<
  width = $( window ).width()-$( window ).width()/12*4;
  height = $( window ).height()-108;
  // Create a svg canvas
  svg = d3.select("#Box1")
    .append("svg")
    .attr("id","model_area")
    .attr("width", width)
    .attr("height", height)
    .on("mousemove",handleMouseMove)
    .on("click",function(){
      var coords = d3.mouse(this);
      if(the_clicked_item != "null"){//有物件正在被選取，先取消它的選取
        if(the_clicked_item.substring(0,4)!="path"){
          if(coords[0] < getTranslateX(d3.select("#"+the_clicked_item).attr("transform"))
            || coords[0] > getTranslateX(d3.select("#"+the_clicked_item).attr("transform"))+ItemWidth
            || coords[1] < getTranslateY(d3.select("#"+the_clicked_item).attr("transform"))
            || coords[1] > getTranslateY(d3.select("#"+the_clicked_item).attr("transform"))+ItemHeight){
            unselectItem();
            closeStatus();
            the_clicked_item="null";
          }
        }else{
          click_path++;
          if(click_path%2 == 1){
            unselectPath();
            the_clicked_item="null";
          }
        }
        
      }
    });
  //add in 0816 with d3 for mv >>>>>
  d3.select("body")
    .on("keydown",function(){
      if(d3.event.keyCode == 46){//delete
        if(the_clicked_item != "null"){
          //東西要先設定乾淨才能清掉
          if(the_clicked_item.substring(0,4)!="path"){
            var fin_paths = d3.select("#"+the_clicked_item).attr("fin_path").split(" ");var i;
            for(i=1;i<fin_paths.length;i++){
              removePath(fin_paths[i]);
            }
            
            var fout_paths = d3.select("#"+the_clicked_item).attr("fout_path").split(" ");
            for(i=1;i<fout_paths.length;i++){
              removePath(fout_paths[i]);
            }
            d3.select("#"+the_clicked_item).remove();
          }else{
            removePath(the_clicked_item);
          }
        }
      }
      if(d3.event.keyCode == 27){//esc
        if(holding_line %2 ==1){
          holding_line = 0;
          _lineFrom="null";
          d3.select("#vmline").remove();
        }
      }
    });
}

function conv_showAdvance_click(){
      conv_showAdvance++;
      if(conv_showAdvance%4 == 2){
        d3.select("#div_convAdvance").style("display","");
      }
      else{
        d3.select("#div_convAdvance").style("display","none");
      }
}

function div_linearAdvance_click(){
	linear_showAdvance++;
    if(linear_showAdvance%4 == 2){
      d3.select("#div_linearAdvance").style("display","");
    }
    else{
      d3.select("#div_linearAdvance").style("display","none");
    }
}

function removePath(the_path){
	var lineFrom_fouts = d3.select("#"+d3.select("#"+the_path).attr("start")).attr("fout").split(" ");
	var lineFrom_foutpaths = d3.select("#"+d3.select("#"+the_path).attr("start")).attr("fout_path").split(" ");
	var lineTo_fins = d3.select("#"+d3.select("#"+the_path).attr("end")).attr("fin").split(" ");
	var lineTo_finpaths = d3.select("#"+d3.select("#"+the_path).attr("end")).attr("fin_path").split(" ");

	var i;var new_fout="";var new_fout_path="";var new_fin="";var new_fin_path="";
	for(i=0;i<lineFrom_fouts.length;i++){
		if(lineFrom_fouts[i] != d3.select("#"+the_path).attr("end")){
			if(i!=0)new_fin += " ";
			new_fout += lineFrom_fouts[i];
		}
	}for(i=0;i<lineFrom_foutpaths.length;i++){
		if(lineFrom_foutpaths[i] != the_path){
			if(i!=0)new_fin += " ";
			new_fout_path += lineFrom_foutpaths[i];
		}
	}for(i=0;i<lineTo_fins.length;i++){
		if(lineTo_fins[i] != d3.select("#"+the_path).attr("start")){
			if(i!=0)new_fin += " ";
			new_fin += lineTo_fins[i];
		}
	}for(i=0;i<lineTo_finpaths.length;i++){
		if(lineTo_finpaths[i] != the_path){
			if(i!=0)new_fin += " ";
			new_fin_path += lineTo_finpaths[i];
		}
	}

	d3.select("#"+d3.select("#"+the_path).attr("start")).attr("fout",new_fout);
	d3.select("#"+d3.select("#"+the_path).attr("start")).attr("fout_path",new_fout_path);
	d3.select("#"+d3.select("#"+the_path).attr("end")).attr("fin",new_fin);
	d3.select("#"+d3.select("#"+the_path).attr("end")).attr("fin_path",new_fin_path);
	d3.select("#"+the_path).remove();	
}

function saveItemParameters(){
	if(the_clicked_item != "null"){
		if(the_clicked_item.substring(0,5) == "Input"){
			d3.select("#"+the_clicked_item).attr("channel",$("#the_inputChannel").val().trim());
      d3.select("#"+the_clicked_item).attr("size",$("#the_inputSize").val().trim());
		}
		else if(the_clicked_item.substring(0,4) == "Conv"){
			d3.select("#"+the_clicked_item).attr("kernal",$("#the_kernalSize").val().trim());
			d3.select("#"+the_clicked_item).attr("filtern",$("#the_filterNum").val().trim());
      d3.select("#"+the_clicked_item).attr("stride",$("#the_stride").val().trim());
      d3.select("#"+the_clicked_item).attr("padding",$("#the_padding").val().trim());
      d3.select("#"+the_clicked_item).attr("delation",$("#the_delation").val().trim());
      d3.select("#"+the_clicked_item).attr("groups",$("#the_groups").val().trim());
		}
		else if(the_clicked_item.substring(0,4) == "Pool"){
			d3.select("#"+the_clicked_item).attr("pools",$("#the_poolSize").val().trim());
		}
		else if(the_clicked_item.substring(0,6) == "Linear"){
			d3.select("#"+the_clicked_item).attr("outf",$("#the_outFeature").val().trim());
		}
		else if(the_clicked_item.substring(0,7) == "Dropout"){
			d3.select("#"+the_clicked_item).attr("toDrop",$("#the_toDrop").val().trim());
		}
	}
}

function showStatus(){
	closeStatus();
	//在Status上設定value
	var intruduction;
	if(the_clicked_item.substring(0,5) == "Input"){
		intruduction = "Layer that represents a particular input port in the network."
		if(d3.select("#"+the_clicked_item).attr("channel") != "null"){
      $("#the_inputChannel").val(d3.select("#"+the_clicked_item).attr("channel"));
		}
    if(d3.select("#"+the_clicked_item).attr("size") != "null"){
      $("#the_inputSize").val(d3.select("#"+the_clicked_item).attr("size"));
    }
		d3.select("#div_inputParameters").style("display","");
		d3.select("#savebtn").style("display","");
	}
	else if(the_clicked_item.substring(0,6) == "Output"){
		intruduction = "Layer that represents a particular output port in the network.";
	}
	else if(the_clicked_item.substring(0,4) == "Conv"){
		intruduction = "Convolution operator for filtering windows of two-dimensional inputs.";
		if(d3.select("#"+the_clicked_item).attr("kernal") != "null"){
	      $("#the_kernalSize").val(d3.select("#"+the_clicked_item).attr("kernal"))
		}
		if(d3.select("#"+the_clicked_item).attr("filtern") != "null"){
	      $("#the_filterNum").val(d3.select("#"+the_clicked_item).attr("filtern"))
		}
	    if(d3.select("#"+the_clicked_item).attr("stride") != "null"){
	      $("#the_stride").val(d3.select("#"+the_clicked_item).attr("stride"))
	    }
	    if(d3.select("#"+the_clicked_item).attr("padding") != "null"){
	      $("#the_padding").val(d3.select("#"+the_clicked_item).attr("padding"))
	    }
	    if(d3.select("#"+the_clicked_item).attr("delation") != "null"){
	      $("#the_delation").val(d3.select("#"+the_clicked_item).attr("delation"))
	    }
	    if(d3.select("#"+the_clicked_item).attr("groups") != "null"){
	      $("#the_groups").val(d3.select("#"+the_clicked_item).attr("groups"))
	    }
		d3.select("#div_convParameters").style("display","");
		d3.select("#savebtn").style("display","");

		if(conv_showAdvance%4==2){
			d3.select("#div_convAdvance").style("display","");
		}
	}
	else if(the_clicked_item.substring(0,4) == "Pool"){
		intruduction = "Max pooling operation for spatial data. ";
		if(d3.select("#"+the_clicked_item).attr("pools") != "null"){
      $("#the_poolSize").val(d3.select("#"+the_clicked_item).attr("pools"))
		}
		d3.select("#div_poolParameters").style("display","");
		d3.select("#savebtn").style("display","");
	}
	else if(the_clicked_item.substring(0,4) == "Relu"){
		intruduction = "An activation function defined as the positive part of its argument.";
	}
	else if(the_clicked_item.substring(0,6) == "Linear"){
		intruduction = "Applies a linear transformation to the incoming data.";
		if(d3.select("#"+the_clicked_item).attr("outf") != "null"){
      		$("#the_outFeature").val(d3.select("#"+the_clicked_item).attr("outf"))
		}
		d3.select("#div_linearParameters").style("display","");
		d3.select("#savebtn").style("display","");

		if(linear_showAdvance%4==2){
			d3.select("#div_linearAdvance").style("display","");
		}
	}
	else if(the_clicked_item.substring(0,7) == "Dropout"){
		intruduction = "Applies dropout to the input.";
		if(d3.select("#"+the_clicked_item).attr("toDrop") != "null"){
      $("#the_toDrop").val(d3.select("#"+the_clicked_item).attr("toDrop"))
		}
		d3.select("#div_dropoutParameters").style("display","");
		d3.select("#savebtn").style("display","");
	}
	$("#the_intruduction").text(intruduction);
	d3.select("#the_intruduction").style("display","");
}

function closeStatus(){
    $("#the_intruduction").text("");
  	d3.select("#savebtn").style("display","none");
  	d3.select("#the_intruduction").style("display", "none");
    $("#the_kernalSize").val("");
    $("#the_filterNum").val("");
    $("#the_stride").val("");
    $("#the_padding").val("");
    $("#the_delation").val("");
    $("#the_groups").val("");
    $("#the_poolSize").val("");
    $("#the_outFeature").val("");
    $("#the_toDrop").val("");
    $("#the_inputChannel").val("");
    $("#the_inputSize").val("");


    d3.select("#div_inputParameters").style("display","none");
    d3.select("#div_convParameters").style("display","none");
    d3.select("#div_poolParameters").style("display","none");
    d3.select("#div_linearParameters").style("display","none");
    d3.select("#div_dropoutParameters").style("display","none");

    d3.select("#div_convAdvance").style("display","none");
    d3.select("#div_linearAdvance").style("display","none");
}

function selectItem(the_id){
	if(the_clicked_item != "null"){//有物件正在被選取，先取消它的選取
  		d3.select("#"+the_clicked_item).attr("clicked","n");
  		if(the_clicked_item != "null"){//有物件正在被選取，先取消它的選取
      		if(the_clicked_item.substring(0,4)!="path"){
      			unselectItem();
      		}
      		else{
      			unselectPath();
      		}
      	}
  	}

	//讓這個物件看起來被選取
	d3.select("#"+the_id).attr("clicked","y");
    d3.select("#"+the_id+"_rect").style({
      stroke: '#00FF66',
        'stroke-width': 5,
        opacity: 1
    });
    the_clicked_item = the_id;


    //在Status上設定value
    showStatus();
}

function selectPath(pathID){
	if(the_clicked_item != "null"){//有物件正在被選取，先取消它的選取
  		if(the_clicked_item.substring(0,4)!="path"){
  			unselectItem();
  			closeStatus();
  		}
  		else{
  			unselectPath();
  		}
  	}
				   	
	//讓這個物件看起來被選取
	d3.select("#"+pathID).attr("clicked","y");
  d3.select("#"+pathID).style({
    stroke: '#00FF66',
    'stroke-width': 7
  });
  the_clicked_item = pathID;
	click_path=1;

	closeStatus();
}

function unselectItem(){
	d3.select("#"+the_clicked_item).attr("clicked","n");
	d3.select("#"+the_clicked_item+"_rect").style({
	  stroke: 'black',
	    'stroke-width': 2,
	    opacity: .65
	});
}

function unselectPath(){
	d3.select("#"+the_clicked_item).attr("clicked","n");
	d3.select("#"+the_clicked_item).style({
      stroke: '#6986B9',
      'stroke-width': 5
    });
}

function AllowDrop(event){
  event.preventDefault();
}

function Drag(event){
  event.dataTransfer.setData("text",event.currentTarget.id);
}

function Drop(event){//放下
  event.preventDefault();
  var data=event.dataTransfer.getData("text");
  addRect(data,event);//產生元件在畫圖區中
}

function Drop_default(event){
  event.preventDefault();
  var data=event.dataTransfer.getData("text");
  event.currentTarget.appendChild(document.getElementById(data));
}

function Drop_trash(event){
  event.preventDefault();
  var data=event.dataTransfer.getData("text");
  var trash = document.getElementById(data);
  var trash_code = document.getElementById(data+"_code");
  trash.remove();
}

function setCode(_code,num) {//(alt,count)
  var _Box2 = document.getElementById("layer");
  var newbr = document.createElement('br');
  var newCode = document.createTextNode(_code);
  newCode.id = _code+num+"_code";
  newCode.innerHTML = _code;
}

function ChangeBgColor(img_n){
  var img_fileName = document.getElementById(img_n).alt;
  document.getElementById(img_n).src = "pictrue/" + img_fileName+"_mousein.png";
}

function ChangeBgColorBack(img_n){
  var img_fileName = document.getElementById(img_n).alt;
  document.getElementById(img_n).src = "pictrue/" + img_fileName+".png";
}

//add in 0816 with d3 for mv <<<<<<<<<
function addRect(the_type,event){
  //new Rect
  var newg = svg.append("g");
  var bachgroundColor;
  var rectWidth=ItemWidth;
  var rectHeight=ItemHeight;
  var the_id;
  if(the_type == "Conv"){
    count[0]++;
    the_id=the_type+count[0];
    bachgroundColor = '#006CFF';
    newg.attr("kernal","")
    	.attr("filtern","")
      .attr("stride","1")
      .attr("padding","0")
      .attr("delation","")
      .attr("groups","");
  }else if(the_type == "Pool"){
    count[1]++;
    the_id=the_type+count[1];
    bachgroundColor = '#ff00cc';
    newg.attr("pools","");
  }else if(the_type == "Relu"){
    count[2]++;
    the_id=the_type+count[2];
    bachgroundColor = '#ffa62d';
  }else if(the_type == "Linear"){
    count[3]++;
    the_id=the_type+count[3];
    bachgroundColor = '#ffff66';
    newg.attr("outf","");
  }else if(the_type == "Input"){
    count[4]++;
    the_id=the_type+count[4];
    bachgroundColor = '#f7f7f7';
    if($("#sample_project").val() == "MNIST")
      newg.attr("channel","1");
    else
       newg.attr("channel","3");
    if($("#sample_project").val() == "CIFAR10")
      newg.attr("size","32");
    else
      newg.attr("size","28");
  }else if(the_type == "Output"){
    count[5]++;
    the_id=the_type+count[5];
    bachgroundColor = '#f7f7f7';
  }else if(the_type == "Dropout"){
    count[6]++;
    the_id=the_type+count[6];
    bachgroundColor = '#f7c7f7';
    newg.attr("toDrop","0.2");
  }

  var rectX = event.clientX - document.getElementById('Box3').offsetWidth - (ItemWidth/2);
  var rectY = event.clientY - (ItemHeight/2) - 100;

  //Drag nodes
  var drag = d3.behavior.drag()
      .on("dragstart", function() {
          d3.event.sourceEvent.stopPropagation()
      })
      .on("drag", dragmove);

  
  newg.attr("id",the_id)
  	.attr("fin","null")
    .attr("fout","null")
    .attr("fin_path","null")
    .attr("fout_path","null")
    .attr("clicked","n")
    .attr("transform", "translate(" + rectX + "," + rectY + ")")
    .attr("class", "first")
    .call(drag)
    .on("click", function(){
      if(d3.select("#"+the_id).attr("clicked") == "n"){//這個物件還沒被選取
      	//讓這個物件看起來被選取
      	selectItem(the_id);
      }
    });
  newg.append("rect").attr({
      id: the_id+"_rect",
      width: rectWidth, 
      height: rectHeight,
      rx: 10, 
      ry: 10
    }).style({
      fill: bachgroundColor,
      stroke: 'black',
      'stroke-width': 2,
      opacity: .65,
    });
  newg.append("text").text(the_type)
    .attr("x",rectWidth/3)
    .attr("y",rectHeight/1.5)
    .attr("font-size", "26")
    .attr("font-family", "Calibri")
    .attr("fill","black")
    .attr("padding", "10");
  if(the_type != "Input"){
    newg.append("circle").attr({
      r: rectWidth/(ItemHeight/2),
      cx: rectWidth/2,
      cy: "0"
    }).style("fill", "DarkGray")
      .style("stroke", "black")
      .style("stroke-width", "1")
    .on("mouseover", handleMouseOver)
    .on("mouseout", handleMouseOut)
    .on("click",handlePointClick)
    .attr({
      belong: the_id,
      type: "in"
    });
  }
  if(the_type != "Output"){
    newg.append("circle").attr({
      r: rectWidth/(ItemHeight/2),
      cx: rectWidth/2,
      cy: rectHeight,
    }).style("fill", "DarkGray")
      .style("stroke", "black")
      .style("stroke-width", "1")
    .on("mouseover", handleMouseOver)
    .on("mouseout", handleMouseOut)
    .on("click",handlePointClick)
    .attr({
      belong: the_id,
      type: "out"
    });
  }
}

//Drag handler
function dragmove() {
  var x = d3.event.x - (ItemWidth/2);
  var y = d3.event.y - (ItemHeight/2);
  d3.select(this).attr("transform", "translate(" + x + "," + y + ")");

  var fin_paths = d3.select(this).attr("fin_path").split(" ");
  if(fin_paths.length > 1) {
    for(var i=1;i<fin_paths.length;i++){
      var pathID=fin_paths[i];
        var startX = getdX(d3.select("#"+pathID).attr("d"),"start");
        var startY = getdY(d3.select("#"+pathID).attr("d"),"start");
        var endX = x+(ItemWidth/2);
        var endY = y;
        d3.select("#"+pathID).attr("d",
                        "M "+startX+" "+startY+" "+
                        "Q "+(startX+Math.abs(endX-startX)/8)+" "+(startY+Math.abs(endY-startY)*3/8)+" "+
                        ", "+(startX+(endX-startX)/2)+" "+(startY+(endY-startY)/2)+" "+
                        "T "+endX+" "+endY);
    
    }
  }
  var fout_paths = d3.select(this).attr("fout_path").split(" ");
  if(fout_paths.length > 1) {
    for(var i=1;i<fout_paths.length;i++){
      var pathID=fout_paths[i];
        var startX = x+(ItemWidth/2);
	    var startY = y+ItemHeight;
	    var endX = getdX(d3.select("#"+pathID).attr("d"),"end");
	    var endY = getdY(d3.select("#"+pathID).attr("d"),"end");
	    d3.select("#"+pathID).attr("d",
	                    "M "+startX+" "+startY+" "+
	                    "Q "+(startX+Math.abs(endX-startX)/8)+" "+(startY+Math.abs(endY-startY)*3/8)+" "+
	                    ", "+(startX+(endX-startX)/2)+" "+(startY+(endY-startY)/2)+" "+
	                    "T "+endX+" "+endY);
	  
    }
  }
}
function getdX(str,to_find){
  if(to_find=="start")
    return parseInt(str.split(" ")[1]);
  else
    return parseInt(str.split(" ")[10]);
}
function getdY(str,to_find){
  if(to_find=="start")
    return parseInt(str.split(" ")[2]);
  else
    return parseInt(str.split(" ")[11]);
}

// Create Event Handlers for mouse
function handleMouseOver(d) {  // Add interactivity
  // Use D3 to select element, change color and size
  d3.select(this).attr({
    r: ItemWidth/(ItemHeight/2) * 1.5,
  }).style("fill", "#7cfc00");

  if(holding_line%2 == 1){//holding //防呆
    if(_lineFrom == d3.select(this).attr("belong")){
      d3.select(this).style("fill", "#ff0000");
      d3.select("#vmline").style("stroke", "#ff0000");
      CANNOT_CLICK = 1;
    }
    if(d3.select(this).attr("type")=="out"){
      d3.select(this).style("fill", "#ff0000");
      d3.select("#vmline").style("stroke", "#ff0000");
      CANNOT_CLICK = 1;
    }
  }else if(d3.select(this).attr("type")=="in"){
      d3.select(this).style("fill", "#ff0000");
      CANNOT_CLICK = 1;
  }
}

function handleMouseOut() {
  // Use D3 to select element, change color back to normal
  d3.select(this).attr({
    r: ItemWidth/(ItemHeight/2),
  }).style("fill", "DarkGray ");
  CANNOT_CLICK = 0;
  if(holding_line%2==1){
    d3.select("#vmline").style("stroke", "#6986B9");
  }
}

function handleMouseMove() {
  if(holding_line%2==1){
    var x = d3.event.x - document.getElementById("Box3").offsetWidth-32;
    var y = d3.event.y - document.getElementById("Box1").offsetTop - 108;
    d3.select("#vmline")
    .attr("x2", x)
    .attr("y2", y);
  }
}

function handlePointClick(){
  if(CANNOT_CLICK == 1){
    alert("you can't do this");
  }
  else{
    _lineTo=d3.select(this).attr("belong");
    holding_line++;
    if(holding_line%2==0 && _lineFrom!="null"){//連線 把兩個物件連起來並讓線記錄他的start與end
      var pathID = "path"+path_count;
      var startX = getTranslateX(d3.select("#"+_lineFrom).attr("transform"))+(ItemWidth/2);
      var startY = getTranslateY(d3.select("#"+_lineFrom).attr("transform"))+ItemHeight;
      var endX = getTranslateX(d3.select("#"+_lineTo).attr("transform"))+(ItemWidth/2);
      var endY = getTranslateY(d3.select("#"+_lineTo).attr("transform"));
      var path = svg.append("path")
                    .attr("id",pathID)
                    .style("stroke", "#6986B9")
                    .style("stroke-width", "5")
                    .attr("d",
                          "M "+startX+" "+startY+" "+
                          "Q "+(startX+(endX-startX)/8)+" "+(startY+(endY-startY)*3/8)+" "+
                          ", "+(startX+(endX-startX)/2)+" "+(startY+(endY-startY)/2)+" "+
                          "T "+endX+" "+endY)
                    .attr("fill","none")
                    .attr("start",_lineFrom)
                    .attr("end",_lineTo)
                    .attr("clicked","n")
                .on("click", function(){
              if(d3.select("#"+pathID).attr("clicked") == "n"){//這個物件還沒被選取
                  //讓這個物件看起來被選取
                  selectPath(pathID);
              }
              });
      d3.select("#"+_lineFrom).attr("fout",d3.select("#"+_lineFrom).attr("fout")+" "+_lineTo);//讓start的物件記錄他的fout
      d3.select("#"+_lineFrom).attr("fout_path",d3.select("#"+_lineFrom).attr("fout_path")+" "+pathID);
      d3.select("#"+_lineTo).attr("fin",d3.select("#"+_lineTo).attr("fin")+" "+_lineFrom);//讓end的物件記錄他的fin
      d3.select("#"+_lineTo).attr("fin_path",d3.select("#"+_lineTo).attr("fin_path")+" "+pathID);
      path_count++;
      _lineFrom="null";
      d3.select("#vmline").remove();
    }
    else{//拉線
      _lineFrom=d3.select(this).attr("belong");
      var vmline = svg.append("line")
                    .attr("id","vmline")
                    .style("stroke", "#6986B9")
                    .style("stroke-width", "5")
                    .style("opacity", ".8")
                    .attr("x1", getTranslateX(d3.select("#"+_lineFrom).attr("transform"))+(ItemWidth/2))
                    .attr("y1", getTranslateY(d3.select("#"+_lineFrom).attr("transform"))+ItemHeight)
                    .attr("x2", getTranslateX(d3.select("#"+_lineFrom).attr("transform"))+(ItemWidth/2))
                    .attr("y2", getTranslateY(d3.select("#"+_lineFrom).attr("transform"))+ItemHeight)
                    .attr("start",_lineFrom)
                    .attr("end",_lineFrom)
                    .style("stroke-dasharray", "10");
    }
  }
}

function getTranslateX(tl){
   return parseInt(tl.split(/[(,)]/)[1]);
}
function getTranslateY(tl){
  return parseInt(tl.split(/[(,)]/)[2]);
}

//add in 0816 with d3 for mv >>>>>>>

function chooseDataset(){
  if($("#sample_project").val() != "no" && $("#sample_project").val() != "UPLOAD"){
    d3.select("#moduleUpload_div").style("display","none");
    d3.select("#moduleImg_div").style("display","");
    d3.select("#moduleImg").attr("src","pictrue/"+$("#sample_project").val()+".png");
    DatasetAllocate();
  }
  else if($("#sample_project").val() == "UPLOAD"){
    d3.select("#moduleImg_div").style("display","none");
    d3.select("#moduleImg").attr("src","");

    d3.select("#moduleUpload_div").style("display","");


  }
  else{
    d3.select("#moduleUpload_div").style("display","none");
    d3.select("#moduleImg_div").style("display","none");
    d3.select("#moduleImg").attr("src","");
    $("#training_picNum").val("");
    $("#vaildation_picNum").val("");
    $("#test_picNum").val("");
  }
}
function DatasetAllocate(){
  if($("#sample_project").val() != "no"){
    var total = 0;
    if($("#sample_project").val() == "MNIST")
      total = 70000;
    else if($("#sample_project").val() == "CIFAR10")
      total = 60000;
    else if($("#sample_project").val() == "CATDOG")
      total = 25000;
    else{
      $("#datasetsNum_show_txt").css("display","none");
      $("#custom_input").css("display","");
    }
    if($("#dataset_allocate").val() != "no" && $("#dataset_allocate").val() != "UPLOAD"){
      if($("#dataset_allocate").val() == "custom"){
        $("#datasetsNum_show_txt").css("display","none");
        $("#custom_input").css("display","");
      }
      else{
        $("#datasetsNum_show_txt").css("display","");
        $("#custom_input").css("display","none");
        $("#training_picNum").disabled = true;
        $("#vaildation_picNum").disabled = true;
        $("#test_picNum").disabled = true;
        var train_num = parseInt($("#dataset_allocate").val().split("/")[0]);
        var val_num = parseInt($("#dataset_allocate").val().split("/")[1]);
        var test_num = parseInt($("#dataset_allocate").val().split("/")[2]);
        $("#training_picNum").val(train_num.toString());
        $("#vaildation_picNum").val(val_num.toString());
        $("#test_picNum").val(test_num.toString());
        $("#training_picNum_txt").text(train_num.toString());
        $("#vaildation_picNum_txt").text(val_num.toString());
        $("#test_picNum_txt").text(test_num.toString());
      }
    }
    else{
      $("#training_picNum").val("");
      $("#vaildation_picNum").val("");
      $("#test_picNum").val("");
    }
  }
}

function generateSample_NNmodule(){
  $("#model_area").empty();
  if($("#sample_project").val() == "MNIST"){
  	if($("#sampleNN").val() == "SAMPLE"){
  		addRect_sample("Input","translate(57,12)","1","28","null","null");
	    addRect_sample("Conv","translate(68,135)","5","16","1","2");
	    addRect_sample("Relu","translate(75,216)","null","null","null","null");
	    addRect_sample("Pool","translate(83,302)","2","null","null","null");
	    addRect_sample("Conv","translate(344,223)","5","32","1","2");
	    addRect_sample("Relu","translate(344,303)","null","null","null","null");
	    addRect_sample("Pool","translate(345,391)","2","null","null","null");
	    addRect_sample("Linear","translate(637,327)","1568","10","null","null");
	    addRect_sample("Output","translate(649,406)","null","null","null","null");

	    connect_sample("Input"+count[4],"Conv"+(count[0]-1));
	    connect_sample("Conv"+(count[0]-1),"Relu"+(count[2]-1));
	    connect_sample("Relu"+(count[2]-1),"Pool"+(count[1]-1));
	    connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]));
	    connect_sample("Conv"+(count[0]),"Relu"+(count[2]));
	    connect_sample("Relu"+(count[2]),"Pool"+(count[1]));
	    connect_sample("Pool"+(count[1]),"Linear"+(count[3]));
	    connect_sample("Linear"+(count[3]),"Output"+(count[5]));
  	}
    else if($("#sampleNN").val() == "LeNet"){
    	addRect_sample("Input","translate(57,12)","1","28","null","null");
	    addRect_sample("Conv","translate(68,135)","3","6","1","1");
	    addRect_sample("Pool","translate(75,216)","2","null","null","null");
	    addRect_sample("Conv","translate(344,183)","5","12","1","1");
	    addRect_sample("Pool","translate(351,261)","2","null","null","null");
	    addRect_sample("Linear","translate(620,220)","null","120","null","null");
	    addRect_sample("Linear","translate(627,300)","null","84","null","null");
	    addRect_sample("Linear","translate(634,380)","null","10","null","null");
	    addRect_sample("Output","translate(649,500)","null","null","null","null");

	    connect_sample("Input"+count[4],"Conv"+(count[0]-1));
	    connect_sample("Conv"+(count[0]-1),"Pool"+(count[1]-1));
	    connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]));
	    connect_sample("Conv"+(count[0]),"Pool"+(count[1]));
	    connect_sample("Pool"+(count[1]),"Linear"+(count[3]-2));
	    connect_sample("Linear"+(count[3]-2),"Linear"+(count[3]-1));
	    connect_sample("Linear"+(count[3]-1),"Linear"+(count[3]));
	    connect_sample("Linear"+(count[3]),"Output"+(count[5]));
    }
    else if($("#sampleNN").val() == "AlexNet"){

    }
    else if($("#sampleNN").val() == "VGG"){
    	alert("尚未建立");
    }
    else if($("#sampleNN").val() == "ResNet"){
    	alert("尚未建立");
    }
  }
  else if($("#sample_project").val() == "CIFAR10"){
  	if($("#sampleNN").val() == "SAMPLE"){
		  addRect_sample("Input","translate(57,12)","3","32","null","null");
	    addRect_sample("Conv","translate(68,135)","5","6","1","0");
	    addRect_sample("Relu","translate(75,216)","null","null","null","null");
	    addRect_sample("Pool","translate(83,302)","2","null","null","null");
	    addRect_sample("Conv","translate(344,223)","5","16","1","0");
	    addRect_sample("Relu","translate(344,303)","null","null","null","null");
	    addRect_sample("Pool","translate(345,391)","2","null","null","null");
	    addRect_sample("Linear","translate(637,327)","400","120","null","null");
	    addRect_sample("Relu","translate(640,418)","null","null","null","null");
	    addRect_sample("Linear","translate(643,508)","120","84","null","null");
	    addRect_sample("Relu","translate(646,598)","null","null","null","null");
	    addRect_sample("Linear","translate(649,688)","84","10","null","null");
	    addRect_sample("Output","translate(652,778)","null","null","null","null");

	    connect_sample("Input"+count[4],"Conv"+(count[0]-1));
	    connect_sample("Conv"+(count[0]-1),"Relu"+(count[2]-3));
	    connect_sample("Relu"+(count[2]-3),"Pool"+(count[1]-1));
	    connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]));
	    connect_sample("Conv"+(count[0]),"Relu"+(count[2]-2));
	    connect_sample("Relu"+(count[2]-2),"Pool"+(count[1]));
	    connect_sample("Pool"+(count[1]),"Linear"+(count[3]-2));
	    connect_sample("Linear"+(count[3]-2),"Relu"+(count[2]-1));
	    connect_sample("Relu"+(count[2]-1),"Linear"+(count[3]-1));
	    connect_sample("Linear"+(count[3]-1),"Relu"+(count[2]));
	    connect_sample("Relu"+(count[2]),"Linear"+(count[3]));
	    connect_sample("Linear"+(count[3]),"Output"+(count[5]));
  	}
    else if($("#sampleNN").val() == "LeNet"){
      addRect_sample("Input","translate(57,12)","3","32","null","null");
      addRect_sample("Conv","translate(68,135)","5","6","1","0");
      addRect_sample("Relu","translate(75,216)","null","null","null","null");
      addRect_sample("Pool","translate(83,302)","2","null","null","null");
      addRect_sample("Conv","translate(344,223)","5","16","1","0");
      addRect_sample("Relu","translate(344,303)","null","null","null","null");
      addRect_sample("Pool","translate(345,391)","2","null","null","null");
      addRect_sample("Linear","translate(637,327)","400","120","null","null");
      addRect_sample("Relu","translate(640,418)","null","null","null","null");
      addRect_sample("Linear","translate(643,508)","120","84","null","null");
      addRect_sample("Relu","translate(646,598)","null","null","null","null");
      addRect_sample("Linear","translate(649,688)","84","10","null","null");
      addRect_sample("Output","translate(652,778)","null","null","null","null");

      connect_sample("Input"+count[4],"Conv"+(count[0]-1));
      connect_sample("Conv"+(count[0]-1),"Relu"+(count[2]-3));
      connect_sample("Relu"+(count[2]-3),"Pool"+(count[1]-1));
      connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]));
      connect_sample("Conv"+(count[0]),"Relu"+(count[2]-2));
      connect_sample("Relu"+(count[2]-2),"Pool"+(count[1]));
      connect_sample("Pool"+(count[1]),"Linear"+(count[3]-2));
      connect_sample("Linear"+(count[3]-2),"Relu"+(count[2]-1));
      connect_sample("Relu"+(count[2]-1),"Linear"+(count[3]-1));
      connect_sample("Linear"+(count[3]-1),"Relu"+(count[2]));
      connect_sample("Relu"+(count[2]),"Linear"+(count[3]));
      connect_sample("Linear"+(count[3]),"Output"+(count[5]));
    }
    else if($("#sampleNN").val() == "AlexNet"){

    }
    else if($("#sampleNN").val() == "VGG"){
    	alert("尚未建立");
    }
    else if($("#sampleNN").val() == "ResNet"){
    	alert("尚未建立");
    }
  }
  else if($("#sample_project").val() == "CATDOG"){
    alert("尚未建立");
  }
  else if($("#sample_project").val() == "UPLOAD"){
    if($("#sampleNN").val() == "SAMPLE"){
      addRect_sample("Input","translate(31,41)","3","28","null","null");
      addRect_sample("Conv","translate(32,138)","3","64","1","1");
      addRect_sample("Relu","translate(35,215)","null","null","null","null");
      addRect_sample("Conv","translate(40,293)","3","64","1","1");
      addRect_sample("Relu","translate(44,365)","null","null","null","null");
      addRect_sample("Pool","translate(48,452)","2","null","null","null");

      addRect_sample("Conv","translate(277,159)","3","128","1","1");
      addRect_sample("Relu","translate(283,243)","null","null","null","null");
      addRect_sample("Conv","translate(295,320)","3","128","1","1");
      addRect_sample("Relu","translate(306,395)","null","null","null","null");
      addRect_sample("Pool","translate(320,475)","2","null","null","null");

      addRect_sample("Conv","translate(495,193)","3","256","1","1");
      addRect_sample("Relu","translate(518,265)","null","null","null","null");
      addRect_sample("Conv","translate(530,340)","3","256","1","1");
      addRect_sample("Relu","translate(542,416)","null","null","null","null");
      addRect_sample("Conv","translate(550,493)","3","256","1","1");
      addRect_sample("Relu","translate(560,571)","null","null","null","null");
      addRect_sample("Pool","translate(573,645)","2","null","null","null");

      addRect_sample("Conv","translate(721,220)","3","512","1","1");
      addRect_sample("Relu","translate(736,295)","null","null","null","null");
      addRect_sample("Conv","translate(746,369)","3","512","1","1");
      addRect_sample("Relu","translate(757,448)","null","null","null","null");
      addRect_sample("Conv","translate(767,522)","3","512","1","1");
      addRect_sample("Relu","translate(779,594)","null","null","null","null");
      addRect_sample("Pool","translate(791,669)","2","null","null","null");

      addRect_sample("Linear","translate(936,251)","null","1024","null","null");
      addRect_sample("Relu","translate(952,322)","null","null","null","null");
      addRect_sample("Linear","translate(960,395)","null","1024","null","null");
      addRect_sample("Relu","translate(974,468)","null","null","null","null");
      addRect_sample("Linear","translate(986,543)","null","2","null","null");

      addRect_sample("Output","translate(1005,628)","null","null","null","null");


      connect_sample("Input"+count[4],"Conv"+(count[0]-9));
      connect_sample("Conv"+(count[0]-9),"Relu"+(count[2]-11));
      connect_sample("Relu"+(count[2]-11),"Conv"+(count[0]-8));
      connect_sample("Conv"+(count[0]-8),"Relu"+(count[2]-10));
      connect_sample("Relu"+(count[2]-10),"Pool"+(count[1]-3));

      connect_sample("Pool"+(count[1]-3),"Conv"+(count[0]-7));
      connect_sample("Conv"+(count[0]-7),"Relu"+(count[2]-9));
      connect_sample("Relu"+(count[2]-9),"Conv"+(count[0]-6));
      connect_sample("Conv"+(count[0]-6),"Relu"+(count[2]-8));
      connect_sample("Relu"+(count[2]-8),"Pool"+(count[1]-2));

      connect_sample("Pool"+(count[1]-2),"Conv"+(count[0]-5));
      connect_sample("Conv"+(count[0]-5),"Relu"+(count[2]-7));
      connect_sample("Relu"+(count[2]-7),"Conv"+(count[0]-4));
      connect_sample("Conv"+(count[0]-4),"Relu"+(count[2]-6));
      connect_sample("Relu"+(count[2]-6),"Conv"+(count[0]-3));
      connect_sample("Conv"+(count[0]-3),"Relu"+(count[2]-5));
      connect_sample("Relu"+(count[2]-5),"Pool"+(count[1]-1));
      
      connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]-2));
      connect_sample("Conv"+(count[0]-2),"Relu"+(count[2]-4));
      connect_sample("Relu"+(count[2]-4),"Conv"+(count[0]-1));
      connect_sample("Conv"+(count[0]-1),"Relu"+(count[2]-3));
      connect_sample("Relu"+(count[2]-3),"Conv"+(count[0]-0));
      connect_sample("Conv"+(count[0]-0),"Relu"+(count[2]-2));
      connect_sample("Relu"+(count[2]-2),"Pool"+(count[1]));

      connect_sample("Pool"+(count[1]),"Linear"+(count[3]-2));
      connect_sample("Linear"+(count[3]-2),"Relu"+(count[2]-1));
      connect_sample("Relu"+(count[2]-1),"Linear"+(count[3]-1));
      connect_sample("Linear"+(count[3]-1),"Relu"+(count[2]));
      connect_sample("Relu"+(count[2]),"Linear"+(count[3]));
      connect_sample("Linear"+(count[3]),"Output"+(count[5]));
    }
    else if($("#sampleNN").val() == "AlexNet"){
      addRect_sample("Input","translate(57,12)","3","28","null","null");
      addRect_sample("Conv","translate(68,135)","5","16","1","2");
      addRect_sample("Relu","translate(75,216)","null","null","null","null");
      addRect_sample("Pool","translate(83,302)","2","null","null","null");
      addRect_sample("Conv","translate(344,223)","5","32","1","2");
      addRect_sample("Relu","translate(344,303)","null","null","null","null");
      addRect_sample("Pool","translate(345,391)","2","null","null","null");
      addRect_sample("Linear","translate(637,327)","1568","2","null","null");
      addRect_sample("Output","translate(649,406)","null","null","null","null");

      connect_sample("Input"+count[4],"Conv"+(count[0]-1));
      connect_sample("Conv"+(count[0]-1),"Relu"+(count[2]-1));
      connect_sample("Relu"+(count[2]-1),"Pool"+(count[1]-1));
      connect_sample("Pool"+(count[1]-1),"Conv"+(count[0]));
      connect_sample("Conv"+(count[0]),"Relu"+(count[2]));
      connect_sample("Relu"+(count[2]),"Pool"+(count[1]));
      connect_sample("Pool"+(count[1]),"Linear"+(count[3]));
      connect_sample("Linear"+(count[3]),"Output"+(count[5]));
    }
  }
  else{
  	alert("Please choose a dataset first.");
  	$("#sampleNN").val("no");
  }
}

function show_result_page(){
  if(do_canvas == 1){
    setTimeout(function(){
      $("#iframe").css("width","100%");
      $.get("result_page_accuracy.html",function(data){ //初始將a.html include div#iframe
        $("#iframe").html(data);
      });
      $("#iframe_loss").css("width","100%");
      $.get("result_page_loss.html",function(data){ //初始將a.html include div#iframe
        $("#iframe_loss").html(data);
      });
    }, 1000);
  }
  // $('body').loading({
  //   stoppable: false
  // });
  // setTimeout(function() {
  //   javascript:location.href='result_page.html';
  // }, 2000);
}

function readURL(input){
  if(input.files && input.files[0]){
    var imageTagID = input.getAttribute("targetID");
    var reader = new FileReader();
    reader.onload = function (e) {
      var img = document.getElementById(imageTagID);
      img.setAttribute("src", e.target.result)
    }
    reader.readAsDataURL(input.files[0]);
  }
}

function chooseDataset_change(){
  if($("#chooseDataset").val() == "upload"){
    $("#upload_dataset_div").css("display","");
  }
  else{
    $("#upload_dataset_div").css("display","none");
  }
}

function dataset_start_inference(){
  if($("#chooseDataset").val() == "upload"){
    alert("[error] The function is not finish yet.");
  }
  else{

    var getString = 0;
    $.get('python/result.txt')
    .done(function (data) {
      var num = data.split("{").length-2;
      var result = data.split("{");
      var expect = new Array(num);
      var predict = new Array(num);

      for(var i=0;i<num;i++){
        expect[i] = result[i+1].split(",")[0];
        predict[i] = result[i+1].split(",")[1];
      }
      expect[num] = "Total:";
      predict[num] = result[num+1].substring(0,4);
      show_testResult(expect,predict);

      getString = 1;
      $("#dataset_inference_selector").css("display","none");
      $("#dataset_inference_show_result").css("display","");
    })
    .fail(function (jqXHR, textStatus, errorThrown){ 
      var err = textStatus + ", " + errorThrown; 
      console.log( "Request Failed: " + err ); 
      alert("[error] Please start the training first.");
    });
  }
}

function show_testResult(expect,predict){
  for(var i=0;i<expect.length-1;i++){
    $("#dataset_inference_show_result_content").append(
      '<div style="margin-top:50px;overflow: hidden;"> <div class="col-md-4"> <img style="height: 80px;" id="sample_test'+i+'_pic" src="python/Image_result/Image'+i+'.jpg"/> </div> <div class="col-md-1"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;">'+expect[i]+'</div> <div class="col-md-3"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;">'+predict[i]+'</div> </div>'
    );
  }
  $("#dataset_inference_show_result_content").append(
      '<div style="margin-top:50px;overflow: hidden;"> <div class="col-md-4"> <img style="height: 80px;" id="sample_test'+(expect.length-1)+'_pic" src="pictrue/last_pic.png"/> </div> <div class="col-md-1"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;">'+expect[expect.length-1]+'</div> <div class="col-md-3"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;">'+predict[expect.length-1]+'</div> </div>'
    );
  $("#dataset_inference_show_result_content").append(
  	'<div id="dataset_inference_show_result_error_content"></div>'
  	);
}

function show_testError_clk(){
  count_show_resultError_clk++;
  if(count_show_resultError_clk%4==2){
    var getString = 0;
    $.get('python/result_error.txt')
    .done(function (data) {
      var num = 10;
      if(data.split("{").length-1 < 10){
        num = data.split("{").length-1;
      }
      var result = data.split("{");
      var expect = new Array(num);
      var predict = new Array(num);

      for(var i=0;i<num;i++){
        expect[i] = result[i+1].split(",")[0];
        predict[i] = result[i+1].split(",")[1];
      }
      show_testResult_error(expect,predict);

      getString = 1;
      $("#dataset_inference_selector").css("display","none");
      $("#dataset_inference_show_result").css("display","");
    })
    .fail(function (jqXHR, textStatus, errorThrown){ 
      var err = textStatus + ", " + errorThrown; 
      console.log( "Request Failed: " + err ); 
      alert("[error]");
    });
  }
  else{
    $("#dataset_inference_show_result_error_content").empty();
  }
}

function show_testResult_error(expect,predict){
  for(var i=0;i<expect.length;i++){
    $("#dataset_inference_show_result_error_content").append(
      '<div style="margin-top:50px;overflow: hidden;"> <div class="col-md-4"> <img style="height: 80px;" id="sample_test'+i+'_pic" src="python/Image_result_error/Image'+i+'.jpg"/> </div> <div class="col-md-1"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;">'+expect[i]+'</div> <div class="col-md-3"></div> <div class="col-md-1" style="font-size:25px;text-align:center;line-height:80px;border-bottom-style: solid;color:red;">'+predict[i]+'</div> </div>'
    );
  }
}

function dataset_inference_back(){
  $("#dataset_inference_show_result").css("display","none");
  $("#dataset_inference_show_result_content").empty();
  $("#dataset_inference_show_result_error_content").empty();
  $("#dataset_inference_selector").css("display","");
}

function result_detail_clk(){
  //show the weight and bias
  $.ajax({
    url: "php/read_result_image.php",
        type: "POST",
        datatype: "html",
        success: function(output) 
        {
            $("#detail_images").html(output);
        },
        error : function()
        {
           alert( "Request failed.\n" );
        }
  });

  //show the feature maps
  $.ajax({
    url: "php/read_result_featureMaps.php",
        type: "POST",
        datatype: "html",
        success: function(output) 
        {
            $("#featureMaps_div").html(output);
        },
        error : function()
        {
        }
  });
}

function show_result_detail(name){
  $("#detail_images").append(
    '<img style="height: 250px;" src="python/weight_bias/'+name+'"/>'
  );
}

function show_result_featureMaps(name){
  var name_num = parseInt(name.substring(5,name.length-4));
  if(name_num == 0){
    $("#featureMaps_div").append(
      '<div class="col-md-2"><img style="width: 70%;margin-top: 70px;" src="python/Feature_Map/image0.jpg"/></div>'
    );
  }
  else{
    if(name_num%10==1){
        $("#featureMaps_div").append(
          '<div class="col-md-1" id="featureMaps_conv'+(parseInt(name_num/10)+1)+'"><font size="5" color="" style="">Conv'+(parseInt(name_num/10)+1)+'</font></div>'
        );
    }
    if(name_num%10==0){
        $("#featureMaps_conv"+(parseInt(name_num/10)+1)).append(
          '<img style="width: 51%;margin-top: 1px;" src="python/Feature_Map/'+name+'"/>'
        );
    }
    else{
        $("#featureMaps_conv"+(parseInt(name_num/10)+1)).append(
          '<img style="width: 51%;margin-top: 1px;" src="python/Feature_Map/'+name+'"/>'
        );
    }
  }
}

function shuffle_click(){
	shuffle_count++;
	if(shuffle_count%4 == 2){
		shuffle = "true";
	}
	else{
	    shuffle = "false";
	}
}

function check_to_run(){
  if(CAN_RUN == 1){
    run();
  }
  else{
    alert("[error] Please check whether the model is ok.");
  }
}



function check_can_run(){
  var id_array = ["Conv","Pool","Relu","Linear","Input","Output","Dropout"];
  CAN_RUN = 1;
  for(var i=0;i<7;i++){
    for(var j=1;j<=count[i];j++){
      if($("#"+id_array[i]+j).length){
        switch (id_array[i]){
          case "Conv":            
            if($("#"+id_array[i]+j).attr("kernal") == ""
                || $("#"+id_array[i]+j).attr("filtern") == ""
                || $("#"+id_array[i]+j).attr("fin").split(" ").length <= 1
                || $("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Pool":
            if($("#"+id_array[i]+j).attr("pools") == ""
                || $("#"+id_array[i]+j).attr("fin").split(" ").length <= 1
                || $("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Relu":
            if($("#"+id_array[i]+j).attr("fin").split(" ").length <= 1
                || $("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Linear":
            if($("#"+id_array[i]+j).attr("outf") == ""
                || $("#"+id_array[i]+j).attr("fin").split(" ").length <= 1
                || $("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Input":
            if($("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Output":
            if($("#"+id_array[i]+j).attr("fin").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
          case "Dropout":
            if($("#"+id_array[i]+j).attr("fin").split(" ").length <= 1
                || $("#"+id_array[i]+j).attr("fout").split(" ").length <= 1)
            {
              CAN_RUN = 0;
            }
            break;
        }
      }
    }
  }
}

function stopclk(){
  do_canvas = 1;
  $.ajax
    ({
        url: "php/user_left.php",
        
        type: "POST",
        datatype: "html",
        success: function(output) 
        {
        	alert("has been reset!");
        	document.getElementById("btn").disabled = false;
        	$("#btn").css("box-shadow","5px 6px 5px 0px #cccccc");
        	document.getElementById("stopbtn").disabled = true;
        	$("#stopbtn").css("box-shadow","0px 2px 1px 0px #cccccc");
            //$("#show").html(output);
        },
        error : function()
        {
           alert( "Request failed.\n");
        }
    });
}

function setTrainBtnAble(){
    document.getElementById("btn").disabled = false;
    $("#btn").css("box-shadow","5px 6px 5px 0px #cccccc");
    document.getElementById("stopbtn").disabled = true;
    $("#stopbtn").css("box-shadow","0px 2px 1px 0px #cccccc");
}

function setTrainBtnDisable(){
    document.getElementById("btn").disabled = true;
    $("#btn").css("box-shadow","0px 2px 1px 0px #cccccc");
    document.getElementById("stopbtn").disabled = false;
    $("#stopbtn").css("box-shadow","5px 6px 5px 0px #cccccc");
}

function upload_my_py_pkl_clk(){
  $('#test_py_uploader').css('display','');
  $('#upload_my_py_pkl_btn').css('display','none')
}

window.addEventListener("load", main, false);