var total_dataset;

function run()
{
    var last_con_filter_num;
    var last_linear_value = 0;
    var linear_value;
    var json_str = '';
    var json_str_data = '';
    var json_str_hy = '';
    alert("start trainning !");

    var current_state = $("#Input"+count[4]).attr("fout");
    last_con_filter_num = $("#Input"+count[4]).attr("channel");
    current_state = current_state.split(" ")[1];

    json_str_data += $("#sample_project").val();

    if(json_str_data == "MNIST")
        linear_value = 28;
    else if(json_str_data == "CIFAR10")
        linear_value = 32;
    else if(json_str_data == "UPLOAD")//承佑2018/11/18加的
        linear_value = 28;

    json_str_data += ',';
    json_str_data += $("#training_picNum").val();
    json_str_data += ',';
    json_str_data += $("#vaildation_picNum").val();
    json_str_data += ',';
    json_str_data += $("#test_picNum").val();
    json_str_data += ',';
    json_str_data += shuffle;


//hyperparameter
    json_str_hy += $("#batch_size").val();
    json_str_hy += ',';
    json_str_hy += $("#lr").val();
    json_str_hy += ',';
    json_str_hy += $("#momentan").val();
    json_str_hy += ',';
    json_str_hy += $("#epoch").val();
    json_str_hy += ',';
    json_str_hy += $("#optimizer").val();


    while(current_state.substring(0,6) != "Output"){
        if(current_state.substring(0,4) == "Conv"){
            json_str += '{';
            json_str += '[';
            json_str += current_state.substring(0,4);
            json_str += '(';

            var cs = $('#' + current_state);
            var kernalSize = cs.attr("kernal");
            var filterNum = cs.attr("filtern");
            var stride = cs.attr("stride");
            var padding = cs.attr("padding");

            json_str += last_con_filter_num + ',';
            json_str += filterNum + ',';
            json_str += kernalSize + ',';
            json_str += stride + ',';
            json_str += padding;

            last_con_filter_num = filterNum;
            linear_value = (linear_value - parseInt(kernalSize) + 2 * parseInt(padding)) / parseInt(stride) + 1;
            linear_value = parseInt(linear_value);

            alert("conv_value: " + linear_value);

            if(linear_value <= 0 ){
                alert("[error] The conv_value is nagtive or less than zero: " + linear_value);
                return;
            }
        }


        else if(current_state.substring(0,4) == "Pool"){

            json_str += '[';
            json_str += current_state.substring(0,4);
            json_str += '(';

            var poolSize = $('#' + current_state).attr("pools");
            json_str += poolSize;
            linear_value /= parseInt(poolSize);
            linear_value = parseInt(linear_value);
        }

        else if(current_state.substring(0,6) == "Linear"){
            json_str += '{';
            json_str += '[';
            json_str += current_state.substring(0,6);
            json_str += '(';

            var inFeature;
            var outFeature = $('#' + current_state).attr("outf");

            if(last_linear_value == 0){
                inFeature = linear_value * linear_value * parseInt(last_con_filter_num);
                last_linear_value = parseInt(outFeature);
            }
            else{
                inFeature = last_linear_value;
                last_linear_value = parseInt(outFeature);
            }
            alert("Linear inFeature_value:" + inFeature);
            if(inFeature <= 0){
                alert("gradient vanishing is been detect , check the conveolution value");
                return;
            }

            json_str += inFeature;
            json_str += ',';
            json_str += outFeature;
        }

        else if(current_state.substring(0,4) == "Relu"){
            json_str += '[';
            json_str += current_state.substring(0,4);
            json_str += '(';
        }
        current_state = $('#' + current_state).attr("fout");
        current_state = current_state.split(" ")[1];
    }

    json_str += '{';
    json_str += '[';
    json_str += current_state.substring(0,6);
    json_str += '(';

    //alert(json_str);
        /*
    var txt = $("#txt").val();
    var cmd_ = "";
   
    var batch_size_ = $("#batch_size").val();
    var learning_rate_ = $("#learning_rate").val();
    var momentan_ = $("#momentan").val();
    var weight_decay_ = $("#weight_decay").val();
    var criterion_ = $("#criterion").val();
    var optimizer_ = $("#optimizer").val();

    cmd_ += "print('batch size: " + batch_size_ + " ')" + '\n';
    cmd_ += "print('learning rate: " + learning_rate_ + " ')" + '\n';
    cmd_ += "print('momentan: " + momentan_ + " ')" + '\n';
    cmd_ += "print('weight decay: " + weight_decay_ + " ')" + '\n';
    cmd_ += "print('criterion: " + criterion_ + " ')" + '\n';
    cmd_ += "print('optimizer: " + optimizer_ + " ')" + '\n';
    */
    //alert("json_str_hy: " + json_str_hy);

    batch_size_frame = parseInt($("#batch_size").val());
    epoch_frame = parseInt($("#epoch").val());
    training_picNum_frame = parseInt($("#training_picNum").val());


    setTrainBtnDisable();

    total_dataset = Math.ceil(epoch_frame*training_picNum_frame/batch_size_frame)-1;
    
    $.ajax
    ({
        url: "php/control.php",
        data: {get_str_data: json_str_data,get_str_model: json_str, get_str_hy: json_str_hy},

        type: "POST",
        datatype: "html",
        success: function(output) 
        {
            setTrainBtnAble();
            $("#config").html(output);
        },
        error : function()
        {
            alert( "Request failed.\n" );
            setTrainBtnAble();
        }
    });
}

