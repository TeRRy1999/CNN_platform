<?php

if ($handle = opendir('C:\\xampp\\htdocs\\CNN_Project\\python\\weight_bias')) { 
  while (false !== ($file = readdir($handle))) {  
  //避免搜尋到的資料夾名稱是false,像是0  
    if ($file != "." && $file != "..") {  
    //去除掉..跟.  
      echo "<script>show_result_detail('$file');</script>";         
    }  
  }  
  closedir($handle);  
}
else{

}
?>