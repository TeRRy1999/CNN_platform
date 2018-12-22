<?php



$stop = 1;

$myfile = fopen("C:\\xampp\\htdocs\\CNN_Project\\python\\stop.txt", "w") or die("Unable to open file!");


fwrite($myfile, $stop);


fclose($myfile);



shell_exec('rmdir C:\\xampp\\htdocs\\CNN_Project\\python\\weight_bias /s /q');
shell_exec('rmdir C:\\xampp\\htdocs\\CNN_Project\\python\\Image_result /s /q');
//$temp = shell_exec('python C:\\xampp\\htdocs\\CNN_Project\\php\\python.py 2>&1');

//print("<div><pre>" . $get_str . "</pre></div>");

?>