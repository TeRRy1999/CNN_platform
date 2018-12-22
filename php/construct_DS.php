<?php    
class MNIST{
    public $content="";
    public $get_str;
    public $Conv_counter=0;
    public $ReLu_counter=0;
    public $Pool_counter=0;
    public $Linear_counetr=0;
    public $Train_data=0;
    public $Validation_data=0;
    public $Test_data=0;
    public $Shuffle=0;
    public $Train_size=0;
    public $Validation_size=0;
    public $Test_size=0;
    public $Batch_size = 0;
    public $learning_ratio = 0.0;
    public $Optimizer;
    public $Epoch;
    public $Show;
    function writelibrary(){    
    $this->content=$this->content."import os\n".
    "import torch\n".
    "import json\n".
    "import torch.nn as nn\n".
    "import torch.utils.data as Data\n".
    "import torchvision\n".
    "import matplotlib.pyplot as plt\n".
    "import torchvision.transforms as transforms\n".
    "import sys\n".
    "jsonData = []\n".
    "jsonData1 = []\n".
    "jsonData2 = []\n".
    "jsonData3 = []\n".
    "import os\n".
    "import random\n".
    "import numpy as np\n".
    "import time\n".
    "import matplotlib\n".
    "import shutil\n".
    "from torch.autograd import Variable\n".
    "from PIL import Image\n"
    ;
    }

    function write_stop(){
        $this->content=$this->content."file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
        "file_object.write('0')\n";
    }


    function writeHyper_Parameters(){
    
        $this->content=$this->content."EPOCH = ".$this->Epoch."\n".              
        "BATCH_SIZE =".$this->Batch_size."\n".
        "LR =".$this->learning_ratio."\n".              
        "DOWNLOAD_MNIST = False\n".
        "DOWNLOAD_CIFAR10 = False\n".
        "DOWNLOAD_CATDOG = False\n".
        "train_count =". $this->Train_data."\n".
        "train_size =".$this->Train_size. "\n".
        "if(train_count%BATCH_SIZE == 0):\n".
        "\ttrain_size = int(train_count/BATCH_SIZE)\n".
        "else:\n".
        "\ttrain_size = int(train_count/BATCH_SIZE)+1\n".
        "print('train_size:',train_size)\n"
        ;
    }

    function vertify_show(){
        if($this->Train_size>10000 && $this->Train_size<=50000){
            $this->Show = $this->Train_size / 250 ;
        }
        elseif($this->Train_size>5000 && $this->Train_size<=10000){
            $this->Show = $this->Train_size / 200 ;
        }

        elseif($this->Train_size>1000 && $this->Train_size<=5000){
            $this->Show = $this->Train_size / 150 ;
        }

        elseif($this->Train_size>500 && $this->Train_size<=1000){
            $this->Show = $this->Train_size / 100 ;
        }

        elseif($this->Train_size>100 && $this->Train_size<=500){
            $this->Show = $this->Train_size / 50 ;
        }

        elseif($this->Train_size> 20 && $this->Train_size<=100){
            $this->Show = $this->Train_size / 20 ;
        }
        else{
            $this->Show = 1; 
        }

    }

    function writeMnist_digits_dataset(){
            $this->content=$this->content."Show = ".(int)$this->Show."\n".
            "Show = int(Show)\n".
            "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias'):\n".
            "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "else:\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result'):\n".
            "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
            "else:\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
            "if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):\n".
            "\tDOWNLOAD_MNIST = True\n".
            "train_data = torchvision.datasets.MNIST(\n".
            "\troot='./mnist/',\n".
            "\ttrain=True,\n".
            "\ttransform=torchvision.transforms.ToTensor(),\n".
            "\tdownload=DOWNLOAD_MNIST,)\n".
            "test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)\n";
            if( $this->Shuffle==1){
                $this->content=$this->content."train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)\n".
                "test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)\n";
            }
            else{
                $this->content=$this->content."train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)\n".
                "test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)\n";
            }
            
            $this->content=$this->content.
            "test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:".$this->Test_data."]/255.\n".
            "test_y = test_data.test_labels[:".$this->Test_data."]\n".
            "print_test_x = test_data.test_data[:".$this->Test_data."]\n".
            "validation_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[".$this->Test_data.":".($this->Test_data+$this->Validation_data)."]/255.\n".
            "validation_y = test_data.test_labels[".$this->Test_data.":".($this->Test_data+$this->Validation_data)."]\n".
            "rnds = [random.randint(0,".($this->Test_data-5).") for _ in range(".($this->Test_data-5).")]\n"
            ;           
        
    }

    
    function write_cnn(){
        $this->content=$this->content."class CNN(nn.Module):\n".
        "\tdef __init__(self):\n".
        "\t\tsuper(CNN, self).__init__()\n"        
        ;
    }

    function write_net(){
        $example1 = explode("{",$this->get_str);
        
        foreach ($example1 as $element1){
            $example2=explode("[",$element1);
            foreach($example2 as $element2) {
                $example3=explode("(",$element2);

                if(trim($example3[0])=="Conv"){
                    $this->Conv_counter++;
                    if($this->Conv_counter>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    $times=0;
                    $example3[1]=trim($example3[1]);
                    $arr=explode(",",$example3[1]);
                    $this->content=$this->content."\t\tself.conv".$this->Conv_counter."= nn.Sequential(\n".
                    "\t\t\tnn.Conv2d(";
                    foreach($arr as $ele){
                        if($times==0){
                            $this->content=$this->content.$ele;
                        }
                        else{
                            $this->content=$this->content.",".$ele;
                        }
                        $times++;
                    }
                    $this->content=$this->content."),\n";
                }


                elseif(trim($example3[0])=="Relu"){
                    $this->ReLu_counter++;
                    $this->content=$this->content."\t\t\tnn.ReLU(),\n";
                }

                elseif(trim($example3[0])=="Pool"){
                    $this->Pool_counter++;
                    $this->content=$this->content."\t\t\tnn.MaxPool2d(".trim($example3[1])."),\n";
                }
                elseif(trim($example3[0])=="Linear"){
                   
                    $this->Linear_counetr++;
                    if($this->Linear_counetr>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    else{
                        if($this->Conv_counter>0){
                            $this->content=$this->content."\t\t\t)\n";
                        }
                    }
                    $times=0;
                    $this->content=$this->content."\t\tself.Linear".$this->Linear_counetr."= nn.Sequential(\n".
                    "\t\t\tnn.Linear(";
                    $arr=explode(",",$example3[1]);
                    foreach($arr as $ele){
                    if($times==0){
                        $this->content=$this->content.$ele;
                    }
                    else{
                        $this->content=$this->content.",".$ele;
                    }
                    $times++;
                }
                $this->content=$this->content."),\n";
            }
        }
    }
    $this->content=$this->content."\t\t\t)\n";
    $this->content=$this->content."\tdef forward(self, x):\n";
        for($i=1;$i<=$this->Conv_counter;$i++){
            $this->content=$this->content."\t\tx = self.conv".$i."(x)\n";
        }
        $this->content=$this->content."\t\tx = x.view(x.size(0), -1)\n";

        for($i=1;$i<$this->Linear_counetr;$i++){
            $this->content=$this->content."\t\tx = self.Linear".$i."(x)\n";
        }
        $this->content=$this->content."\t\toutput = self.Linear".$this->Linear_counetr."(x)\n";
        $this->content=$this->content."\t\treturn output\n".
        "cnn = CNN()\n".
        "print(cnn)\n";

}

    function write_test(){
        $this->Conv_counter=0;
        $this->Linear_counetr=0;
        $this->Pool_counter=0;
        $this->ReLu_counter=0;
        $this->content="";
        $this->writelibrary();
        $this->write_cnn();
        $this->write_net();
        $this->content=$this->content."pict = Image.open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\test.jpg')\n".
        "pict = pict.resize( (28, 28), Image.BILINEAR )\n".
        "pict = pict.convert('L')\n".
        "img = np.array(pict)\n".
        "data = torch.from_numpy(img)\n".
        "test_x = torch.unsqueeze(data , dim=0).type(torch.FloatTensor)\n".
        "test_x = torch.unsqueeze(test_x , dim=1).type(torch.FloatTensor)\n".
        "cnn.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "test_output = cnn(test_x)\n".
        "pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()\n".
        "strs = str(pred_y)\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Test_Result.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "print(\"預測結果: \",pred_y)\n"
        ;
    }

    function write_end(){
        
        if($this->Optimizer=='ADAM'){
            $this->content=$this->content."optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)\n";
        }
        else{
            $this->content=$this->content."optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)\n";
        }
        $this->content=$this->content."loss_func = nn.CrossEntropyLoss()\n".
        "from matplotlib import cm\n".
        "try: from sklearn.manifold import TSNE; HAS_SK = True\n".
        "except: HAS_SK = False; print('Please install sklearn for layer visualization')\n".

 
        "epoch_for_json = 0\n".
        "epoch_for_json1 = 0\n".
        "epoch_for_json2 = 0\n".
        "epoch_for_json3 = 0\n".
        "for epoch in range(EPOCH):\n".
        "\tcnt_ratio = 0\n".
        "\tfor step, (b_x, b_y) in enumerate(train_loader):\n".
        "\t\tif(cnt_ratio > train_size):\n".
        "\t\t\tbreak\n".
        "\t\tcnt_ratio+=1\n".
        "\t\toutput = cnn(b_x)\n".
        "\t\t_, train_pred = torch.max(output.data,1)\n".
        "\t\tloss = loss_func(output, b_y)\n".
        "\t\toptimizer.zero_grad()\n".
        "\t\tloss.backward()\n".
        "\t\toptimizer.step()\n".
        "\t\trunning_correct = torch.sum(train_pred == b_y.data)\n".
        "\t\ttrain_accuracy = float(running_correct) / float(BATCH_SIZE)\n".
        "\t\tif( ((step % Show) == 0) or (step == (train_size-1))  ):\n".
        "\t\t\tprint('Epoch: ', epoch, step,'| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % train_accuracy)\n".
        "\t\t\tlst = [epoch_for_json , float(train_accuracy)]\n".
        "\t\t\tlst1 = [epoch_for_json1 , float(loss.data.numpy())]\n".
        "\t\t\tjsonData.append(lst)\n".
        "\t\t\tjsonData1.append(lst1)\n".
        "\t\t\ttest_output = cnn(validation_x)\n".
        "\t\t\tloss = loss_func(test_output, validation_y)\n".
        "\t\t\tpred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()\n".
        "\t\t\taccuracy = float((pred_y == validation_y.data.numpy()).astype(int).sum()) / float(validation_y.size(0))\n".
        "\t\t\tprint('Epoch: ', epoch, step, '| validation loss: %.4f' % loss.data.numpy(), '| validation accuracy: %.2f' % accuracy)\n".
        "\t\t\tprint(epoch_for_json)\n".
        "\t\t\tlst = [epoch_for_json2 , float(accuracy)]\n".
        "\t\t\tlst1 = [epoch_for_json3 , float(loss.data.numpy())]\n".
        "\t\t\tjsonData2.append(lst)\n".
        "\t\t\tjsonData3.append(lst1)\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData1))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData2))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData3))\n".
        "\t\t\tfile_object  = open(\"C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt\", 'r')\n".
        "\t\t\tfile_object = file_object.read()\n".
        "\t\t\tif file_object == '1':\n".
        "\t\t\t\tos.remove('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\generate_json_tmp.py')\n".
        "\t\t\t\tfile_object2 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object2.write('[[0,0]]')\n".
        "\t\t\t\tfile_object3 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object3.write('[[0,0]]')\n".
        "\t\t\t\tfile_object4 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object4.write('[[0,0]]')\n".
        "\t\t\t\tfile_object5 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object5.write('[[0,0]]')\n".
        "\t\t\t\tsys.exit()\n".
        "\t\tepoch_for_json += 1\n".
        "\t\tepoch_for_json1 += 1\n".
        "\t\tepoch_for_json2 += 1\n".
        "\t\tepoch_for_json3 += 1\n".
        "\t\tif((step == (train_size-1))):\n".
        "\t\t\tbreak\n".
        "\t\tif HAS_SK:\n".
        "\t\t\ttsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)\n".
        "def draw_distribution(ax, data, title='Weights Distribution', xlabel='Value', ylabel='Frequency', bins='auto'):\n".
        "\td = data.numpy().reshape(-1)\n".
        "\td_mean = np.mean(d, axis=0)\n".
        "\td_std  = np.std(d, axis=0, ddof=1)\n".
        "\tn, bins, patches = ax.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)\n".
        "\tax.set_xlabel(xlabel)\n".
        "\tax.set_ylabel(ylabel)\n".
        "\tax.set_title(title)\n".
        "\tax.grid()\n".
        "\tmaxfreq = n.max()\n".
        "\tymax = np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10\n".
        "\tax.set_ylim(top=ymax)\n".
        "\ttyaxis = ymax * 0.9\n".
        "\tmaxvalue = bins.max()\n".
        "\tminvalue = bins.min()\n".
        "\ttxaxis = minvalue + (maxvalue - minvalue) * 0.05\n".
        "\tax.text(txaxis, tyaxis, r'$\mu={:6.3f}$'.format(d_mean))\n".
        "\tax.text(txaxis, tyaxis*0.9, r'$\sigma={:6.3f}$'.format(d_std))\n".
        "def show_distribution(model):\n".
        "\tparams=model.state_dict()\n".
        "\teven = True\n".
        "\tplt.subplots_adjust(wspace =1, hspace =1)\n".
        "\tcnts = 0\n".
        "\tcheck = 0\n".
        "\tif os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias'):\n".
        "\t\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\telse:\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\tfor k,v in params.items():\n".
        "\t\tif k.split('.')[-1] == 'weight' or k.split('.')[-1] == 'bias':\n".
        "\t\t\tif even:\n".
        "\t\t\t\tfig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,3), dpi=80)\n".
        "\t\t\t\tax = ax1\n".
        "\t\t\telse:\n".
        "\t\t\t\tax = ax2\n".
        "\t\t\ttitle = k + '.Distribution'\n".
        "\t\t\tdraw_distribution(ax, v.data.cpu(), title)\n".
        "\t\t\teven = not even\n".
        "\t\t\tif((check%2)==1):\n".
        "\t\t\t\tfig.savefig('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias\\\\image'+ str(cnts)+'.jpg')\n".
        "\t\t\t\tcnts+=1\n".
        "\t\t\tcheck+=1\n".
        "class FeatureExtractor(nn.Module):\n".
        "\tdef __init__(self, submodule):\n".
        "\t\tsuper(FeatureExtractor,self).__init__()\n".
        "\t\tself.submodule = submodule\n".
        "\tdef forward(self, x):\n".
        "\t\toutputs = []\n".
        "\t\tfor name, module in self.submodule._modules.items():\n".
        "\t\t\tif name is \"fc\": x = x.view(x.size(0), -1)\n".
        "\t\t\tif(name[0:6] != 'Linear'):\n".
        "\t\t\t\tx = module(x)\n".
        "\t\t\t\toutputs.append(x)\n".
        "\t\treturn outputs\n".
        "show_distribution(cnn)\n".

        "torch.save(cnn.state_dict(), 'C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl')\n".
        "cnn.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)\n".
        "loss_func = nn.CrossEntropyLoss()\n".
        "import matplotlib\n".
        "strs=\"\"\n".
        "test_output= cnn(test_x)\n".
        "pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()\n".
        "cnt = 0\n".
        "accuracy = 0\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "myexactor=FeatureExtractor(cnn)\n".
        "x=myexactor(test_x)\n".
        "matplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(0)+'.jpg', np.transpose(torchvision.utils.make_grid(test_x[0]).numpy(), (1, 2, 0)))\n".
        "for j in range(len(x)):\n".
        "\tfor i in range(10):\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(10*j+i+1)+'.jpg', x[j].data.numpy()[0,i,:,:],cmap='gray')\n".
        "for i in range(rnds[0],rnds[0]+5):\n".
        "\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result\\\\image'+ str(cnt)+'.jpg', test_data.test_data[i].numpy(),cmap='gray')\n".
        "\tstrs+=('{'+str(test_y.data.numpy()[i])+','+str(pred_y[i]))\n".
        "\taccuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))\n".
        "\tcnt+=1\n".
        "strs+=('{'+str(accuracy))\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "strs=\"\"\n".
        "cnt = 0\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "for i in range(".$this->Test_data."):\n".
        "\tif pred_y[i] != test_y.data.numpy()[i]:\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error\\\\image'+ str(cnt)+'.jpg', print_test_x[i].numpy(),cmap='gray')\n".
        "\t\tstrs+=('{'+str(test_y.data.numpy()[i])+','+str(pred_y[i]))\n".
        "\t\tcnt+=1\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result_error.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
        "file_object.write('1')\n"
        ;
    }
}

class UPLOAD{
    public $content="";
    public $get_str;
    public $Conv_counter=0;
    public $ReLu_counter=0;
    public $Pool_counter=0;
    public $Linear_counetr=0;
    public $Shuffle=0;
    public $Batch_size = 0;
    public $learning_ratio = 0.0;
    public $Optimizer;
    public $Epoch;
    public $Show;

    function writelibrary(){    
        $this->content=$this->content."import os\n".
        "import torch\n".
        "import json\n".
        "import torch.nn as nn\n".
        "import torch.utils.data as Data\n".
        "import torchvision\n".
        "import matplotlib.pyplot as plt\n".
        "import torchvision.transforms as transforms\n".
        "import sys\n".
        "jsonData = []\n".
        "jsonData1 = []\n".
        "jsonData2 = []\n".
        "jsonData3 = []\n".
        "from torchvision import datasets, transforms\n".
        "import os\n".
        "import random\n".
        "import numpy as np\n".
        "import time\n".
        "import matplotlib\n".
        "import shutil\n".
        "from torch.autograd import Variable\n".
        "from PIL import Image\n".
        "from torchvision.datasets import ImageFolder\n".
        "from torchvision import transforms as T\n".
        "import torch as t\n".
        "from torch.utils import data\n".
        "import zipfile\n"
        ;
        }
    function write_stop(){
        $this->content=$this->content."file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
        "file_object.write('0')\n";
    }

    function writeHyper_Parameters(){
    
        $this->content=$this->content."EPOCH = ".$this->Epoch."\n".              
        "BATCH_SIZE =".$this->Batch_size."\n".
        "LR =".$this->learning_ratio."\n".              
        "DOWNLOAD_MNIST = False\n".
        "DOWNLOAD_CIFAR10 = False\n".
        "DOWNLOAD_CATDOG = False\n"
        ;
    }
    function writeMnist_digits_dataset(){
        $this->content=$this->content."if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_data'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_data')\n".        
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\valid_data'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\valid_data')\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\test_data'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\test_data')\n".
        "fileUnZip = zipfile.ZipFile('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\train.zip')\n".
        "fileUnZip.extractall('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_data')\n".
        "fileUnZip.close()\n".
        "fileUnZip = zipfile.ZipFile('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\valid.zip')\n".
        "fileUnZip.extractall('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\valid_data')\n".
        "fileUnZip.close()\n".
        "fileUnZip = zipfile.ZipFile('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\test.zip')\n".
        "fileUnZip.extractall('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\test_data')\n".
        "fileUnZip.close()\n".
        "file_path_train = \"\"\n".
        "file_path_valid= \"\"\n".
        "file_path_test= \"\"\n".
        "for dirName,sub_dirNames,fileNames in os.walk('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_data'):\n".
        "\tif(len(sub_dirNames)>1):\n".
        "\t\tfile_path_train += dirName\n".
        "for dirName,sub_dirNames,fileNames in os.walk('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\valid_data'):\n".
        "\tif(len(sub_dirNames)>1):\n".
        "\t\tfile_path_valid += dirName\n".
        "for dirName,sub_dirNames,fileNames in os.walk('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\test_data'):\n".
        "\tif(len(sub_dirNames)>1):\n".
        "\t\tfile_path_test += dirName\n".
        "transform = T.Compose([\n".
        "\tT.Scale((28,28)),\n".
        "\tT.ToTensor(),\n".
        "])\n".
        "print(file_path_train)\n".
        "print(file_path_valid)\n".
        "print(file_path_test)\n".
        "dataset = ImageFolder(file_path_train,transform = transform)\n".
        "vali_dataset = ImageFolder(file_path_valid,transform = transform)\n".
        "test_dataset = ImageFolder(file_path_test,transform = transform)\n".
        "trainn_data = len(dataset)\n".
        "example_classes = test_dataset.classes\n".
        "print(example_classes)\n".
        "print(dataset[0][0].size())\n".
        "print(\"Train data number\",len(dataset)) \n".
        "print(\"valid data number\",len(vali_dataset))\n".
        "print(\"Test data number\",len(test_dataset))\n".
        "if(trainn_data%BATCH_SIZE == 0):\n".
        "\tTrain_size = int(trainn_data/BATCH_SIZE)\n".
        "else:\n".
        "\tTrain_size = int(trainn_data/BATCH_SIZE)+1\n".
        "Show = 0\n".
        "if(Train_size>10000 and Train_size<=50000):\n".
        "\tShow = Train_size / 250\n".
        "elif(Train_size>5000 and Train_size<=10000):\n".
        "\tShow = Train_size / 200\n".
        "elif(Train_size>1000 and Train_size<=5000):\n".
        "\tShow = Train_size / 150\n".
        "elif(Train_size>500 and Train_size<=1000):\n".
        "\tShow = Train_size / 100\n".
        "elif(Train_size>100 and Train_size<=500):\n".
        "\tShow = Train_size / 50\n".
        "elif(Train_size> 20 and Train_size<=100):\n".
        "\tShow = Train_size / 20\n".
        "else:\n".
        "\tShow = 1\n".
        "Show = int(Show)\n"
        ;
    }

    function write_cnn(){
        $this->content=$this->content."class CNN(nn.Module):\n".
        "\tdef __init__(self):\n".
        "\t\tsuper(CNN, self).__init__()\n"        
        ;
    }

    function write_net(){
        $example1 = explode("{",$this->get_str);
        
        foreach ($example1 as $element1){
            $example2=explode("[",$element1);
            foreach($example2 as $element2) {
                $example3=explode("(",$element2);

                if(trim($example3[0])=="Conv"){
                    $this->Conv_counter++;
                    if($this->Conv_counter>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    $times=0;
                    $example3[1]=trim($example3[1]);
                    $arr=explode(",",$example3[1]);
                    $this->content=$this->content."\t\tself.conv".$this->Conv_counter."= nn.Sequential(\n".
                    "\t\t\tnn.Conv2d(";
                    foreach($arr as $ele){
                        if($times==0){
                            $this->content=$this->content.$ele;
                        }
                        else{
                            $this->content=$this->content.",".$ele;
                        }
                        $times++;
                    }
                    $this->content=$this->content."),\n";
                }


                elseif(trim($example3[0])=="Relu"){
                    $this->ReLu_counter++;
                    $this->content=$this->content."\t\t\tnn.ReLU(),\n";
                }

                elseif(trim($example3[0])=="Pool"){
                    $this->Pool_counter++;
                    $this->content=$this->content."\t\t\tnn.MaxPool2d(".trim($example3[1])."),\n";
                }
                elseif(trim($example3[0])=="Linear"){
                   
                    $this->Linear_counetr++;
                    if($this->Linear_counetr>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    else{
                        if($this->Conv_counter>0){
                            $this->content=$this->content."\t\t\t)\n";
                        }
                    }
                    $times=0;
                    $this->content=$this->content."\t\tself.Linear".$this->Linear_counetr."= nn.Sequential(\n".
                    "\t\t\tnn.Linear(";
                    $arr=explode(",",$example3[1]);
                    foreach($arr as $ele){
                    if($times==0){
                        $this->content=$this->content.$ele;
                    }
                    else{
                        $this->content=$this->content.",".$ele;
                    }
                    $times++;
                }
                $this->content=$this->content."),\n";
            }
        }
    }
    $this->content=$this->content."\t\t\t)\n";
    $this->content=$this->content."\tdef forward(self, x):\n";
        for($i=1;$i<=$this->Conv_counter;$i++){
            $this->content=$this->content."\t\tx = self.conv".$i."(x)\n";
        }
        $this->content=$this->content."\t\tx = x.view(x.size(0), -1)\n";

        for($i=1;$i<$this->Linear_counetr;$i++){
            $this->content=$this->content."\t\tx = self.Linear".$i."(x)\n";
        }
        $this->content=$this->content."\t\toutput = self.Linear".$this->Linear_counetr."(x)\n";
        $this->content=$this->content."\t\treturn output\n".
        "cnn = CNN()\n".
        "print(cnn)\n";
    }

    function write_test(){
        $this->Conv_counter=0;
        $this->Linear_counetr=0;
        $this->Pool_counter=0;
        $this->ReLu_counter=0;
        $this->content="";
        $this->writelibrary();
        $this->write_cnn();
        $this->write_net();
        $this->content=$this->content."transform = T.Compose([\n".
        "\t\tT.Scale((28,28))\n,".
        "\t\tT.ToTensor(),\n".
        "])\n".
        "file_path_valid= \"\"\n".
        "for dirName,sub_dirNames,fileNames in os.walk('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\valid_data'):\n".
        "\tif(len(sub_dirNames)>1):\n".
        "\t\tfile_path_valid += dirName\n".
        "vali_dataset = ImageFolder(file_path_valid,transform = transform)\n".
        "example_classes = vali_dataset.classes\n".
        "print(example_classes)\n".
        "pict = Image.open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\test.jpg')\n".
        "pict = pict.resize( (28, 28), Image.BILINEAR )\n".
        "pict = pict.convert('RGB')\n".
        "img = np.array(pict)\n".
        "data = torch.from_numpy(img)\n".
        "test_x = torch.unsqueeze(data , dim=0).type(torch.FloatTensor)\n".
        "test_x = test_x.permute(0,3,1,2)\n".
        "print(test_x.size())\n".
        "cnn.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "test_output = cnn(test_x)\n".
        "pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()\n".
        "strs = example_classes[pred_y]\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Test_Result.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "print(\"預測結果: \",pred_y)\n";
    }


    function write_end(){
        $this->content=$this->content."epoch_for_json = 0\n".
        "epoch_for_json1 = 0\n".
        "epoch_for_json2 = 0\n".
        "epoch_for_json3 = 0\n"
        ;
        if($this->Optimizer=='ADAM'){
            $this->content=$this->content."optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)\n";
        }
        else{
            $this->content=$this->content."optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)\n";
        }
        $this->content=$this->content."loss_func = nn.CrossEntropyLoss()\n".
        "train_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)\n".
        "vali_train_loader = Data.DataLoader(dataset=vali_dataset, batch_size=BATCH_SIZE, shuffle=True)\n".
        "testloader =  Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)\n".
        "for epoch in range(EPOCH):\n".
        "\tfor step, (b_x, b_y) in enumerate(train_loader):\n".
        "\t\trunning_correct = 0\n".
        "\t\toutput = cnn(b_x)\n".
        "\t\t_, train_pred = torch.max(output.data,1)\n".
        "\t\tloss = loss_func(output, b_y)\n".
        "\t\toptimizer.zero_grad()\n".
        "\t\tloss.backward()\n".
        "\t\toptimizer.step()\n".
        "\t\trunning_correct = torch.sum(train_pred == b_y.data)\n".
        "\t\ttrain_accuracy = float(running_correct) / float(BATCH_SIZE)\n".
        "\t\tloss_valid_num = 0\n".
        "\t\trunning_correct = 0\n".
        "\t\tif (step % Show == 0 or step == (Train_size-1)):\n".
        "\t\t\tfor cnts, (b_x, b_y) in enumerate(vali_train_loader):\n".
        "\t\t\t\toutput = cnn(b_x)\n".
        "\t\t\t\t_, train_pred = torch.max(output.data,1)\n".
        "\t\t\t\tloss_valid = loss_func(output, b_y)\n".
        "\t\t\t\tloss_valid_num += loss_valid.data.numpy()\n".
        "\t\t\t\trunning_correct += torch.sum(train_pred == b_y.data)\n".
        "\t\t\tvali_train_accuracy = float(running_correct) / float((cnts+1)*BATCH_SIZE)\n".
        "\t\t\tloss_valid_num = float(loss_valid_num) / float(cnts+1)\n".
        "\t\t\tprint('Epoch: ', epoch, step,'| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % train_accuracy)\n".
        "\t\t\tlst = [epoch_for_json , float(train_accuracy)]\n".
        "\t\t\tlst1 = [epoch_for_json1 , float(loss.data.numpy())]\n".
        "\t\t\tjsonData.append(lst)\n".
        "\t\t\tjsonData1.append(lst1)\n".

        "\t\t\tprint('Epoch: ', epoch, step, '| validation loss: %.4f' % loss_valid_num, '| validation accuracy: %.2f' % vali_train_accuracy)\n".
        "\t\t\tprint('\\n')\n".
        "\t\t\tlst = [epoch_for_json2 , float(vali_train_accuracy)]\n".
        "\t\t\tlst1 = [epoch_for_json3 , float(loss_valid_num)]\n".
        "\t\t\tjsonData2.append(lst)\n".
        "\t\t\tjsonData3.append(lst1)\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData1))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData2))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData3))\n".
        "\t\t\tfile_object  = open(\"C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt\", 'r')\n".
        "\t\t\tfile_object = file_object.read()\n".
        "\t\t\tif file_object == '1':\n".
        "\t\t\t\tos.remove('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\generate_json_tmp.py')\n".
        "\t\t\t\tfile_object2 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object2.write('[[0,0]]')\n".
        "\t\t\t\tfile_object3 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object3.write('[[0,0]]')\n".
        "\t\t\t\tfile_object4 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object4.write('[[0,0]]')\n".
        "\t\t\t\tfile_object5 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object5.write('[[0,0]]')\n".
        "\t\t\t\tsys.exit()\n".
        "\t\tepoch_for_json += 1\n".
        "\t\tepoch_for_json1 += 1\n".
        "\t\tepoch_for_json2 += 1\n".
        "\t\tepoch_for_json3 += 1\n".
        "\t\tif((step == (Train_size-1))):\n".
        "\t\t\tbreak\n".

        "def draw_distribution(ax, data, title='Weights Distribution', xlabel='Value', ylabel='Frequency', bins='auto'):\n".
        "\td = data.numpy().reshape(-1)\n".
        "\td_mean = np.mean(d, axis=0)\n".
        "\td_std  = np.std(d, axis=0, ddof=1)\n".
        "\tn, bins, patches = ax.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)\n".
        "\tax.set_xlabel(xlabel)\n".
        "\tax.set_ylabel(ylabel)\n".
        "\tax.set_title(title)\n".
        "\tax.grid()\n".
        "\tmaxfreq = n.max()\n".
        "\tymax = np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10\n".
        "\tax.set_ylim(top=ymax)\n".
        "\ttyaxis = ymax * 0.9\n".
        "\tmaxvalue = bins.max()\n".
        "\tminvalue = bins.min()\n".
        "\ttxaxis = minvalue + (maxvalue - minvalue) * 0.05\n".
        "\tax.text(txaxis, tyaxis, r'$\mu={:6.3f}$'.format(d_mean))\n".
        "\tax.text(txaxis, tyaxis*0.9, r'$\sigma={:6.3f}$'.format(d_std))\n".
        "def show_distribution(model):\n".
        "\tparams=model.state_dict()\n".
        "\teven = True\n".
        "\tplt.subplots_adjust(wspace =1, hspace =1)\n".
        "\tcnts = 0\n".
        "\tcheck = 0\n".
        "\tif os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias'):\n".
        "\t\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\telse:\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\tfor k,v in params.items():\n".
        "\t\tif k.split('.')[-1] == 'weight' or k.split('.')[-1] == 'bias':\n".
        "\t\t\tif even:\n".
        "\t\t\t\tfig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,3), dpi=80)\n".
        "\t\t\t\tax = ax1\n".
        "\t\t\telse:\n".
        "\t\t\t\tax = ax2\n".
        "\t\t\ttitle = k + '.Distribution'\n".
        "\t\t\tdraw_distribution(ax, v.data.cpu(), title)\n".
        "\t\t\teven = not even\n".
        "\t\t\tif((check%2)==1):\n".
        "\t\t\t\tfig.savefig('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias\\\\image'+ str(cnts)+'.jpg')\n".
        "\t\t\t\tcnts+=1\n".
        "\t\t\tcheck+=1\n".
        "show_distribution(cnn)\n".
        "torch.save(cnn.state_dict(), 'C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl')\n".
        "cnn.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "def imshow(img,i,correct):\n".
        "\timg = img / 2 + 0.5\n".
        "\tnpimg = img.numpy()\n".
        "\tplt.imshow(np.transpose(npimg, (1, 2, 0)))\n".
        "\tif (correct == 0):\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result\\\\image'+str(i)+'.jpg',np.transpose(npimg, (1, 2, 0)))\n".
        "\telse:\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error\\\\image'+str(i)+'.jpg',np.transpose(npimg, (1, 2, 0)))\n".
        "class FeatureExtractor(nn.Module):\n".
        "\tdef __init__(self, submodule):\n".
        "\t\tsuper(FeatureExtractor,self).__init__()\n".
        "\t\tself.submodule = submodule\n".
        "\tdef forward(self, x):\n".
        "\t\toutputs = []\n".
        "\t\tfor name, module in self.submodule._modules.items():\n".
        "\t\t\tif name is \"fc\": x = x.view(x.size(0), -1)\n".
        "\t\t\tif(name[0:6] != 'Linear'):\n".
        "\t\t\t\tx = module(x)\n".
        "\t\t\t\toutputs.append(x)\n".
        "\t\treturn outputs\n".
        "cnt = 0\n".
        "show_cnt = 0\n".
        "strs=\"\"\n".
        "strs_show=\"\"\n".
        "feature_cnt = 0\n".
        "with torch.no_grad():\n".
        "\trunning_correct = 0\n".
        "\tfor k, data in enumerate(testloader, 0):\n".
        "\t\timages, labels = data\n".
        "\t\toutputs = cnn(images)\n".
        "\t\t_, predicted = torch.max(outputs, 1)\n".
        "\t\tif (feature_cnt == 0):\n".
        "\t\t\tmyexactor=FeatureExtractor(cnn)\n".
        "\t\t\tx=myexactor(images)\n".
        "\t\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(0)+'.jpg', np.transpose(torchvision.utils.make_grid(images[0]).numpy(), (1, 2, 0)))\n".
        "\t\t\tfor j in range(len(x)):\n".
        "\t\t\t\tfor i in range(10):\n".
        "\t\t\t\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(10*j+i+1)+'.jpg', x[j].data.numpy()[0,i,:,:],cmap='gray')\n".
        "\t\t\tfeature_cnt+=1\n".
        "\t\tfor i in range(BATCH_SIZE):\n".
        "\t\t\tif(show_cnt<=4):\n".
        "\t\t\t\timshow(torchvision.utils.make_grid(images[i]),show_cnt,0)\n".
        "\t\t\t\tshow_cnt+=1\n".
        "\t\t\t\tstrs_show+=('{'+example_classes[labels.data.numpy()[i]]+','+example_classes[predicted.data.numpy()[i]])\n".
        "\t\t\tif(labels.data.numpy()[i]!=predicted[i]):\n".
        "\t\t\t\timshow(torchvision.utils.make_grid(images[i]),cnt,1)\n".
        "\t\t\t\tcnt+=1\n".
        "\t\t\t\tstrs+=('{'+example_classes[labels.data.numpy()[i]]+','+example_classes[predicted.data.numpy()[i]])\n".
        "\t\trunning_correct += torch.sum(predicted == labels.data)\n".
        "\ttext_file = open(\"C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result_error.txt\", \"w\")\n".
        "\ttext_file.write(strs)\n".
        "\ttext_file.close()\n".
        "\ttest_accuracy =  float(running_correct) / float((k+1)*BATCH_SIZE)\n".
        "\tstrs_show+=('{'+str(test_accuracy))\n".
        "\ttext_file = open(\"C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result.txt\", \"w\")\n".
        "\ttext_file.write(strs_show)\n".
        "\ttext_file.close()\n".
        "\tprint('test accuracy: %.2f' % test_accuracy)\n".
        "file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
        "file_object.write('1')\n"
        ;
    }

}



class CIFAR10 {
    public $content="";
    public $get_str;
    public $Conv_counter=0;
    public $ReLu_counter=0;
    public $Pool_counter=0;
    public $Linear_counetr=0;
    public $pre_str;
    public $Train_data=0;
    public $Validation_data=0;
    public $Test_data=0;
    public $Shuffle=0;
    public $Train_size=0;
    public $Validation_size=0;
    public $Test_size=0;
    public $Batch_size = 0;
    public $learning_ratio = 0.0;
    public $Optimizer;
    public $Epoch;
    public $Show;

    function writelibrary(){    
        $this->content=$this->content."import os\n".
        "import torch\n".
        "import json\n".
        "import torch.nn as nn\n".
        "import torch.utils.data as Data\n".
        "import torchvision\n".
        "import torchvision.transforms as transforms\n".
        "import torch.optim as optim\n".
        "import torch.nn.functional as F\n".
        "import matplotlib.pyplot as plt\n".
        "import torchvision.transforms as transforms\n".
        "import sys\n".
        "import numpy as np\n".
        "import shutil\n".
        "from torchvision.transforms import ToPILImage\n".
        "show = ToPILImage()\n".
        "import torchvision as tv\n".
        "from decimal import *\n".
        "jsonData = []\n".
        "jsonData1 = []\n".
        "jsonData2 = []\n".
        "jsonData3 = []\n".
        "import os\n".
        "from PIL import Image\n".
        "import matplotlib\n"
        ;
        }

        function write_stop(){
            $this->content=$this->content."file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
            "file_object.write('0')\n".
            "EPOCH = ".$this->Epoch."\n".
            "train_count = ".$this->Train_data."\n".
            "validation_count =".$this->Validation_data."\n".
            "test_count = ".$this->Test_data."\n".
            "BATCH_SIZE = ".$this->Batch_size."\n".
            "train_size = ".$this->Train_size."\n".
            "validation_size = ".$this->Validation_size."\n".
            "test_size = ".$this->Test_size."\n".
            "if(train_count%BATCH_SIZE == 0):\n".
        "\ttrain_size = int(train_count/BATCH_SIZE)\n".
        "else:\n".
        "\ttrain_size = int(train_count/BATCH_SIZE)+1\n"
            ;
        }

        function vertify_show(){
            if($this->Train_size>10000 && $this->Train_size<=50000){
                $this->Show = $this->Train_size / 250 ;
            }
            elseif($this->Train_size>5000 && $this->Train_size<=10000){
                $this->Show = $this->Train_size / 200 ;
            }
    
            elseif($this->Train_size>1000 && $this->Train_size<=5000){
                $this->Show = $this->Train_size / 150 ;
            }
    
            elseif($this->Train_size>500 && $this->Train_size<=1000){
                $this->Show = $this->Train_size / 100 ;
            }
    
            elseif($this->Train_size>100 && $this->Train_size<=500){
                $this->Show = $this->Train_size / 50 ;
            }
    
            elseif($this->Train_size> 20 && $this->Train_size<=100){
                $this->Show = $this->Train_size / 20 ;
            }
            else{
                $this->Show = 1; 
            }
    
        }
    
    function CIFAR10_dataset(){
        $this->content=$this->content."Show = ".(int)$this->Show."\n".
        "Show = int(Show)\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias'):\n".
            "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "else:\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
            "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result'):\n".
            "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
            "else:\n".
            "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "transform = transforms.Compose(\n".
        "\t[transforms.ToTensor(),\n".
        "\ttransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n";
        $this->content=$this->content."trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)\n".
        "testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)\n";
        if( $this->Shuffle==1){
            $this->content=$this->content."trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)\n".
            "testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)\n";
        }
        else{
            $this->content=$this->content."trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=False, num_workers=0)\n".
            "testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=0)\n";
        }

        $this->content=$this->content."classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')\n"
        ;
    }

    function write_cnn(){
        $this->content=$this->content."class Net(nn.Module):\n".
        "\tdef __init__(self):\n".
        "\t\tsuper(Net, self).__init__()\n"
        ;
    }

    function write_net(){
        $example1 = explode("{",$this->get_str);
        
        foreach ($example1 as $element1){
            $example2=explode("[",$element1);
            foreach($example2 as $element2) {
                $example3=explode("(",$element2);

                if(trim($example3[0])=="Conv"){
                    $this->Conv_counter++;
                    if($this->Conv_counter>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    $times=0;
                    $example3[1]=trim($example3[1]);
                    $arr=explode(",",$example3[1]);
                    $this->content=$this->content."\t\tself.conv".$this->Conv_counter."= nn.Sequential(\n".
                    "\t\t\tnn.Conv2d(";
                    foreach($arr as $ele){
                        if($times==0){
                            $this->content=$this->content.$ele;
                        }
                        else{
                            $this->content=$this->content.",".$ele;
                        }
                        $times++;
                    }
                    $this->content=$this->content."),\n";
                }


                elseif(trim($example3[0])=="Relu"){
                    $this->ReLu_counter++;
                    $this->content=$this->content."\t\t\tnn.ReLU(),\n";
                }

                elseif(trim($example3[0])=="Pool"){
                    $this->Pool_counter++;
                    $this->content=$this->content."\t\t\tnn.MaxPool2d(".trim($example3[1])."),\n";
                }
                elseif(trim($example3[0])=="Linear"){
                   
                    $this->Linear_counetr++;
                    if($this->Linear_counetr>1){
                        $this->content=$this->content."\t\t\t)\n";
                    }
                    else{
                        if($this->Conv_counter>0){
                            $this->content=$this->content."\t\t\t)\n";
                        }
                    }
                    $times=0;
                    $this->content=$this->content."\t\tself.Linear".$this->Linear_counetr."= nn.Sequential(\n".
                    "\t\t\tnn.Linear(";
                    $arr=explode(",",$example3[1]);
                    foreach($arr as $ele){
                    if($times==0){
                        $this->content=$this->content.$ele;
                    }
                    else{
                        $this->content=$this->content.",".$ele;
                    }
                    $times++;
                }
                $this->content=$this->content."),\n";
            }
        }
    }
    $this->content=$this->content."\t\t\t)\n";
    $this->content=$this->content."\tdef forward(self, x):\n";
        for($i=1;$i<=$this->Conv_counter;$i++){
            $this->content=$this->content."\t\tx = self.conv".$i."(x)\n";
        }
        $this->content=$this->content."\t\tx = x.view(x.size(0), -1)\n";

        for($i=1;$i<=$this->Linear_counetr;$i++){
            $this->content=$this->content."\t\tx = self.Linear".$i."(x)\n";
        }
        $this->content=$this->content."\t\treturn x\n"
        ."net = Net()\n"
        ;
}

    function write_test(){
        $this->Conv_counter=0;
        $this->Linear_counetr=0;
        $this->Pool_counter=0;
        $this->ReLu_counter=0;
        $this->content="";
        $this->writelibrary();
        $this->write_cnn();
        $this->write_net();
        $this->content=$this->content."classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')\n".
        "pict = Image.open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\upload\\\\test.jpg')\n".
        "pict = pict.resize( (32, 32), Image.BILINEAR )\n".
        "pict = pict.convert('RGB')\n".
        "img = np.array(pict)\n".
        "data = torch.from_numpy(img)\n".
        "cnn.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "test_x = torch.unsqueeze(data , dim=0).type(torch.FloatTensor)\n".
        "test_x = test_x.permute(0,3,1,2)\n".
        
        "outputs = net(test_x)\n".
        "_, predicted = torch.max(outputs, 1)\n".
        "strs = str(classes[predicted])\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Test_Result.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "print(\"預測結果: \",classes[predicted])\n"
        ;
    }

    function write_end(){
        $this->content=$this->content."criterion = nn.CrossEntropyLoss()\n".
        "optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)\n".
        "epoch_for_json = 0\n".
        "epoch_for_json1 = 0\n".
        "epoch_for_json2 = 0\n".
        "epoch_for_json3 = 0\n".
        "for epoch in range(EPOCH):\n".
        "\trunning_loss = 0.0\n".
        "\tcnt_train = 0\n".
        "\tfor j, data in enumerate(trainloader, 0):\n".
        "\t\tif(cnt_train > train_size):\n".
        "\t\t\tbreak\n".
        "\t\tcnt_train += 1\n".
        "\t\ttrain_class_correct = list(0. for i in range(10))\n".
        "\t\ttrain_class_total = list(0. for i in range(10))\n".
    
        "\t\tinputs, labels = data\n".
        "\t\toptimizer.zero_grad()\n".
        "\t\toutputs = net(inputs)\n".
        "\t\t_, train_pred = torch.max(outputs.data,1)\n".
        "\t\trunning_correct = torch.sum(train_pred == labels.data)\n".
        "\t\ttrain_accuracy = float(running_correct) / float(BATCH_SIZE)\n".
        "\t\tloss = criterion(outputs, labels)\n".
        "\t\tloss.backward()\n".
        "\t\toptimizer.step()\n".
        "\t\trunning_loss += loss.item()\n".
        "\t\tif( ((j % Show) == 0) or (j == (train_size-1) ) ):\n".
        "\t\t\tlst = [epoch_for_json , float(train_accuracy)]\n".
        "\t\t\tlst1 = [epoch_for_json1 , float(loss.data.numpy())]\n".
        "\t\t\tjsonData.append(lst)\n".
        "\t\t\tjsonData1.append(lst1)\n".
        "\t\t\tprint('Epoch: ', epoch , j, '| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.2f' % train_accuracy)\n".

        
        "\t\t\trunning_loss = 0.0\n".
        "\t\t\tclass_correct = list(0. for i in range(10))\n".
        "\t\t\tclass_total = list(0. for i in range(10))\n".
        
        "\t\t\tcnt_ratio = 0\n".
        "\t\t\twith torch.no_grad():\n".
        "\t\t\t\tfor data in testloader:\n".
        "\t\t\t\t\tif(cnt_ratio<validation_size):\n".
        "\t\t\t\t\t\timages, labels = data\n".
        "\t\t\t\t\t\toutputs = net(images)\n".
        "\t\t\t\t\t\t_, predicted = torch.max(outputs, 1)\n".
        "\t\t\t\t\t\tc = (predicted == labels).squeeze()\n".
        "\t\t\t\t\t\tfor i in range(BATCH_SIZE):\n".
        "\t\t\t\t\t\t\tlabel = labels[i]\n".
        "\t\t\t\t\t\t\tclass_correct[label] += c[i].item()\n".
        "\t\t\t\t\t\t\tclass_total[label] += 1\n".
        "\t\t\t\t\t\tcnt_ratio+=1\n".
       
        "\t\t\t\t\telse:\n".
        "\t\t\t\t\t\tbreak\n".
        "\t\t\t\taccuracy=0.0\n".
        "\t\t\t\tfor i in range(10):\n".
        "\t\t\t\t\tif(class_total[i]!=0):\n".
        "\t\t\t\t\t\taccuracy+=class_correct[i] / class_total[i]\n".
        "\t\t\t\taccuracy*=0.1\n".
        "\t\t\t\tloss = criterion(outputs, labels)\n".
        "\t\t\t\tprint('Epoch: ', epoch ,j, '| validation loss: %.4f' % loss.data.numpy(), '| validation accuracy: %.2f' % accuracy)\n".
        "\t\t\t\tlst = [epoch_for_json2 , float(accuracy)]\n".
        "\t\t\t\tlst1 = [epoch_for_json3 , float(loss.data.numpy())]\n".
        "\t\t\t\tjsonData2.append(lst)\n".
        "\t\t\t\tjsonData3.append(lst1)\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData1))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData2))\n".
        "\t\t\twith open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w') as f:\n".
        "\t\t\t\tf.write(json.dumps(jsonData3))\n".
        "\t\t\tfile_object  = open(\"C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt\", 'r')\n".
        "\t\t\tfile_object = file_object.read()\n".
        "\t\t\tif file_object == '1':\n".
        "\t\t\t\tos.remove('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\generate_json_tmp.py')\n".
        "\t\t\t\tfile_object2 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object2.write('[[0,0]]')\n".
        "\t\t\t\tfile_object3 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\train_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object3.write('[[0,0]]')\n".
        "\t\t\t\tfile_object4 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_accuracy.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object4.write('[[0,0]]')\n".
        "\t\t\t\tfile_object5 = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\validation_loss.json', 'w', encoding = 'UTF-8')\n".
        "\t\t\t\tfile_object5.write('[[0,0]]')\n".
        "\t\t\t\tsys.exit()\n".
        "\t\tepoch_for_json += 1\n".
        "\t\tepoch_for_json1 += 1\n".
        "\t\tepoch_for_json2 += 1\n".
        "\t\tepoch_for_json3 += 1\n".
        "\t\tif((j == (train_size-1))):\n".
        "\t\t\tbreak\n".
        "print('Finished Training')\n".
        
        "class FeatureExtractor(nn.Module):\n".
        "\tdef __init__(self, submodule):\n".
        "\t\tsuper(FeatureExtractor,self).__init__()\n".
        "\t\tself.submodule = submodule\n".
        "\tdef forward(self, x):\n".
        "\t\toutputs = []\n".
        "\t\tfor name, module in self.submodule._modules.items():\n".
        "\t\t\tif name is \"fc\": x = x.view(x.size(0), -1)\n".
        "\t\t\tif(name[0:6] != 'Linear'):\n".
        "\t\t\t\tx = module(x)\n".
        "\t\t\t\toutputs.append(x)\n".
        "\t\treturn outputs\n".

        "def draw_distribution(ax, data, title='Weights Distribution', xlabel='Value', ylabel='Frequency', bins='auto'):\n".
        "\td = data.numpy().reshape(-1)\n".
        "\td_mean = np.mean(d, axis=0)\n".
        "\td_std  = np.std(d, axis=0, ddof=1)\n".
        "\tn, bins, patches = ax.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)\n".
        "\tax.set_xlabel(xlabel)\n".
        "\tax.set_ylabel(ylabel)\n".
        "\tax.set_title(title)\n".
        "\tax.grid()\n".
        "\tmaxfreq = n.max()\n".
        "\tymax = np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10\n".
        "\tax.set_ylim(top=ymax)\n".
        "\ttyaxis = ymax * 0.9\n".
        "\tmaxvalue = bins.max()\n".
        "\tminvalue = bins.min()\n".
        "\ttxaxis = minvalue + (maxvalue - minvalue) * 0.05\n".
        "\tax.text(txaxis, tyaxis, r'$\mu={:6.3f}$'.format(d_mean))\n".
        "\tax.text(txaxis, tyaxis*0.9, r'$\sigma={:6.3f}$'.format(d_std))\n".
        "def show_distribution(model):\n".
        "\tparams=model.state_dict()\n".
        "\teven = True\n".
        "\tplt.subplots_adjust(wspace =1, hspace =1)\n".
        "\tcnts = 0\n".
        "\tcheck = 0\n".
        "\tif os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias'):\n".
        "\t\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\telse:\n".
        "\t\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias')\n".
        "\tfor k,v in params.items():\n".
        "\t\tif k.split('.')[-1] == 'weight' or k.split('.')[-1] == 'bias':\n".
        "\t\t\tif even:\n".
        "\t\t\t\tfig, (ax1, ax2) = plt.subplots(1,2,figsize=(9,3), dpi=80)\n".
        "\t\t\t\tax = ax1\n".
        "\t\t\telse:\n".
        "\t\t\t\tax = ax2\n".
        "\t\t\ttitle = k + '.Distribution'\n".
        "\t\t\tdraw_distribution(ax, v.data.cpu(), title)\n".
        "\t\t\teven = not even\n".
        "\t\t\tif((check%2)==1):\n".
        "\t\t\t\tfig.savefig('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\weight_bias\\\\image'+ str(cnts)+'.jpg')\n".
        "\t\t\t\tcnts+=1\n".
        "\t\t\tcheck+=1\n".
        "show_distribution(net)\n".
        "torch.save(net.state_dict(), 'C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl')\n".
        "net.load_state_dict(torch.load('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\params.pkl'))\n".
        "def imshow(img,i,correct):\n".
        "\timg = img / 2 + 0.5\n".
        "\tnpimg = img.numpy()\n".
        "\tplt.imshow(np.transpose(npimg, (1, 2, 0)))\n".
        "\tif (correct == 0):\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result\\\\image'+str(i)+'.jpg',np.transpose(npimg, (1, 2, 0)))\n".
        "\telse:\n".
        "\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error\\\\image'+str(i)+'.jpg',np.transpose(npimg, (1, 2, 0)))\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result_error')\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map')\n".
        "class_correct = list(0. for i in range(10))\n".
        "class_total = list(0. for i in range(10))\n".
        "accuracy=0.0\n".
        "strs=\"\"\n".
        "error_count = 0\n".
        "with torch.no_grad():\n".
        "\tcnt_ratio = 0\n".
        "\tshow = 0\n".
        "\tfeature_cnt = 0\n".
        "\tfor k, data in enumerate(testloader, 0):\n".
        "\t\tif(cnt_ratio < validation_size):\n".
        "\t\t\tcnt_ratio+=1\n".
        "\t\t\tcontinue\n".
        "\t\telif(cnt_ratio > (validation_size+test_size)):\n".
        "\t\t\tbreak\n".
        "\t\telse:\n".
        "\t\t\timages, labels = data\n".
        "\t\t\toutputs = net(images)\n".
        "\t\t\t_, predicted = torch.max(outputs, 1)\n".
        "\t\t\tc = (predicted == labels).squeeze()\n".

        "\t\t\tif (feature_cnt == 0):\n".
        "\t\t\t\tmyexactor=FeatureExtractor(net)\n".
        "\t\t\t\tx=myexactor(images)\n".
        "\t\t\t\timages[0] = images[0] / 2 + 0.5\n".
        "\t\t\t\tnpimg = images[0].numpy()\n".
        "\t\t\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(0)+'.jpg', np.transpose(npimg, (1, 2, 0)))\n".
        "\t\t\t\tfor j in range(len(x)):\n".
        "\t\t\t\t\tfor i in range(10):\n".
        "\t\t\t\t\t\tmatplotlib.image.imsave('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Feature_Map\\\\image'+ str(10*j+i+1)+'.jpg', x[j].data.numpy()[0,i,:,:],cmap='gray')\n".
        "\t\t\t\tfeature_cnt+=1\n".

        "\t\t\tfor i in range(BATCH_SIZE):\n".
        "\t\t\t\tlabel = labels[i]\n".
        "\t\t\t\tclass_correct[label] += c[i].item()\n".
        "\t\t\t\tclass_total[label] += 1\n".
        "\t\t\t\tif (classes[labels[i]] != classes[predicted[i]]):\n".
        "\t\t\t\t\tstrs+=('{'+str(classes[labels[i]])+','+str(classes[predicted[i]]))\n".
        "\t\t\t\t\timshow(torchvision.utils.make_grid(images[i]),error_count,1)\n".
        "\t\t\t\t\terror_count+=1\n".
        "\t\t\tcnt_ratio+=1\n".
        "\t\t\tshow+=1\n".
        "\tfor i in range(10):\n".
        "\t\tif(class_total[i]!=0):\n".
        "\t\t\taccuracy+=class_correct[i] / class_total[i]\n".
        "\taccuracy*=0.1\n".
        "\tprint('Testing Accuracy : %.2f '% (accuracy))\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result_error.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        "if os.path.exists('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result'):\n".
        "\tshutil.rmtree('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        "else:\n".
        "\tos.mkdir('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Image_result')\n".
        
       
        "strs=\"\"\n".
        "testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)\n".
        "dataiter = iter(testloader)\n".
        "for k in range(2):\n".
        "\timages, labels = dataiter.next()\n".
        "\toutputs = net(images)\n".
        "\t_, predicted = torch.max(outputs, 1)\n".
        "\tfor i in range(4):\n".
        "\t\timshow(torchvision.utils.make_grid(images[i]),4*k+i,0)\n".
        "\tfor j in range(4):\n".
        "\t\tstrs+=('{'+str(classes[labels[j]])+','+str(classes[predicted[j]]))\n".
        "\tprint(' '.join('%5s' % classes[labels[j]] for j in range(4)))\n".
        "\tprint(' '.join('%5s' % classes[predicted[j]] for j in range(4)))\n".
        "accuracy = Decimal(accuracy).quantize(Decimal('0.00'))\n".
        "strs+=('{'+str(accuracy))\n".
        "text_file = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\Result.txt', 'w', encoding = 'UTF-8')\n".
        "text_file.write(strs)\n".
        "text_file.close()\n".
        
        "for k, data in enumerate(testloader, 0):\n".
        "\timages, labels = data\n".
        "\toutputs = net(images)\n".
        "\t_, predicted = torch.max(outputs, 1)\n".
        "\tfor i in range(BATCH_SIZE):\n".
        "\t\tif (classes[labels[i]] != classes[predicted[i]]):\n".
        "\t\t\tstrs+=('{'+str(classes[labels[i]])+','+str(classes[predicted[i]]))\n".
        "\t\t\timshow(torchvision.utils.make_grid(images[i]),i*k+i,1)\n".
        
        "file_object = open('C:\\\\xampp\\\\htdocs\\\\CNN_Project\\\\python\\\\stop.txt', 'w', encoding = 'UTF-8')\n".
        "file_object.write('1')\n"
        ;
    }
}
    
    
    function ds_main($get_str,$get_str1,$get_str2){
        //C:\\xampp\\htdocs\\CNN_Project\\python\\
        $file = 'C:\\xampp\\htdocs\\CNN_Project\\python\\generate_json_tmp.py';  // 檔案名稱
        //$file = 'generate_json_tmp.py';
        $file1 = 'C:\\xampp\\htdocs\\CNN_Project\\python\\test.py';  // 檔案名稱
        //$file1 = 'test.py';
        $example1 = explode(",",$get_str);
        if(trim($example1[0])=='MNIST'){
            $example2 = explode(",",$get_str2);
            $Writer=new MNIST();
            $Writer->Validation_data=(int)$example1[2];
            $Writer->Test_data=(int)$example1[3];
            $Writer->Train_data=(int)$example1[1];
            $Writer->Batch_size = (int)$example2[0];
            $Writer->learning_ratio= (double)$example2[1];
            $Writer->Epoch=(int)$example2[3];
            $Writer->Optimizer=trim($example2[4]);
            $Writer->Train_size=$Writer->Train_data / $Writer->Batch_size;
            $Writer->Validation_size=$Writer->Validation_data/$Writer->Batch_size;
            $Writer->Test_size = $Writer->Test_data/$Writer->Batch_size;
            if(trim($example1[4])=='true'){
                $Writer->Shuffle=1;
            }
            else{
                $Writer->Shuffle=0;
            }
            
            $Writer->get_str = $get_str1;
            $Writer->writelibrary();
            $Writer->write_stop();
            $Writer->writeHyper_Parameters();
            $Writer->vertify_show();
            $Writer->writeMnist_digits_dataset();
            $Writer->write_cnn();
            $Writer->write_net();
            $Writer->write_end();
            $fp = fopen($file, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
         
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file success <br/>";
            
            else
                print "write file $file error <br/>";
            $Writer->write_test();
            $fp = fopen($file1, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
             
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file1 success <br/>";
                
            else
                 print "write file $file error <br/>";
            fclose($fp); // 關閉檔案
        }
        else if (trim($example1[0])=='UPLOAD'){
            $example2 = explode(",",$get_str2);
            $Writer = new UPLOAD();
            $Writer->Batch_size = (int)$example2[0];
            $Writer->learning_ratio= (double)$example2[1];
            $Writer->Epoch=(int)$example2[3];
            $Writer->Optimizer=trim($example2[4]);
            if(trim($example1[4])=='true'){
                $Writer->Shuffle=1;
            }
            else{
                $Writer->Shuffle=0;
            }
            $Writer->get_str = $get_str1;
            $Writer->writelibrary();
            $Writer->write_stop();
            $Writer->writeHyper_Parameters();
            $Writer->writeMnist_digits_dataset();
            $Writer->write_cnn();
            $Writer->write_net();
            $Writer->write_end();
            $fp = fopen($file, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
         
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file success <br/>";
            
            else
                print "write file $file error <br/>";
            $Writer->write_test();
            $fp = fopen($file1, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
             
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file1 success <br/>";
                
            else
                 print "write file $file error <br/>";
            fclose($fp); // 關閉檔案
        }

        else if(trim($example1[0])=='CIFAR10'){
            $Writer=new CIFAR10();
            $example2 = explode(",",$get_str2);
            $Writer->Train_data=(int)$example1[1];
            $Writer->Validation_data=((int)$example1[2]);
            $Writer->Test_data=((int)$example1[3]);
            $Writer->Batch_size = (int)$example2[0];
            $Writer->learning_ratio= (double)$example2[1];
            $Writer->Epoch=(int)$example2[3];
            $Writer->Optimizer=trim($example2[4]);
            $Writer->Train_size=$Writer->Train_data / $Writer->Batch_size;
            $Writer->Validation_size=$Writer->Validation_data/$Writer->Batch_size;
            $Writer->Test_size = $Writer->Test_data/$Writer->Batch_size;
            if(trim($example1[4])=='true'){
                $Writer->Shuffle=1;
            }
            else{
                $Writer->Shuffle=0;
            }
            $Writer->get_str = $get_str1;
            $Writer->writelibrary();
            $Writer->write_stop();
            $Writer->vertify_show();
            $Writer->CIFAR10_dataset();
            $Writer->write_cnn();
            $Writer->write_net();
            $Writer->write_end();
            $fp = fopen($file, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
         
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file success <br/>";
            
            else
                print "write file $file error <br/>";

            $Writer->write_test();
            $fp = fopen($file1, "w")  // 開啟檔案
            or exit("file $file open error <br/>");
             
            if (fwrite($fp, $Writer->content))   // 寫入檔案
                print "write file $file1 success <br/>";
                
            else
                 print "write file $file error <br/>";

            fclose($fp); // 關閉檔案
        }
        


    }
//$get_str1="{[Conv(3,6,5,1,0[Relu([Pool(2{[Conv(6,16,5,1,0[Relu([Pool(2{[Linear(400,120[Relu({[Linear(120,84[Relu({[Linear(84,10[Relu(";
    //$get_str="CIFAR10,6000,2000,2000,true";
    //$get_str="MNIST,50000,1000,1000,true";
    //$get_str="UPLOAD,50000,1000,1000,true";
//$get_str1="{[Conv(1,16,5,1,2[Relu([Pool(2{[Conv(16,32,5,1,2[Relu([Pool(2{[Linear(1568,10";   
//$get_str2="50,0.001,0.9,3,ADAM";   
//$get_str="MNIST{[Conv(1,64,3,1,1[Relu({[Conv(64,128,3,1,1[Relu([Pool(2{[Linear(25088,1024[Relu({[Linear(1024,10";
//ds_main($get_str,$get_str1,$get_str2);

