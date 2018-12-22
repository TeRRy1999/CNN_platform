function show(){
	alert("hi");
}

function addRect_sample(the_type,transform,a,b,c,d){
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
    newg.attr("kernal",a)
    	.attr("filtern",b)
      .attr("stride",c)
      .attr("padding",d)
      .attr("delation","")
      .attr("groups","");
  }else if(the_type == "Pool"){
    count[1]++;
    the_id=the_type+count[1];
    bachgroundColor = '#ff00cc';
    newg.attr("pools",a);
  }else if(the_type == "Relu"){
    count[2]++;
    the_id=the_type+count[2];
    bachgroundColor = '#ffa62d';
  }else if(the_type == "Linear"){
    count[3]++;
    the_id=the_type+count[3];
    bachgroundColor = '#ffff66';
    newg.attr("inf",a)
    	.attr("outf",b);
  }else if(the_type == "Input"){
    count[4]++;
    the_id=the_type+count[4];
    bachgroundColor = '#f7f7f7';
    newg.attr("channel",a);
    newg.attr("size",b);
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
    .attr("transform", transform)
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

function connect_sample(from,to){
	var pathID = "path"+path_count;
    var startX = getTranslateX(d3.select("#"+from).attr("transform"))+(ItemWidth/2);
    var startY = getTranslateY(d3.select("#"+from).attr("transform"))+ItemHeight;
    var endX = getTranslateX(d3.select("#"+to).attr("transform"))+(ItemWidth/2);
    var endY = getTranslateY(d3.select("#"+to).attr("transform"));
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
                  .attr("start",from)
                  .attr("end",to)
                  .attr("clicked","n")
		          .on("click", function(){
				    if(d3.select("#"+pathID).attr("clicked") == "n"){//這個物件還沒被選取
				      	//讓這個物件看起來被選取
				        selectPath(pathID);
				    }
		    	  });
    d3.select("#"+from).attr("fout",d3.select("#"+from).attr("fout")+" "+to);//讓start的物件記錄他的fout
    d3.select("#"+from).attr("fout_path",d3.select("#"+from).attr("fout_path")+" "+pathID);
    d3.select("#"+to).attr("fin",d3.select("#"+to).attr("fin")+" "+from);//讓end的物件記錄他的fin
    d3.select("#"+to).attr("fin_path",d3.select("#"+to).attr("fin_path")+" "+pathID);
    path_count++;
}