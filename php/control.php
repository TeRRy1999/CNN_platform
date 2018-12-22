<?php

include 'construct_DS.php';

$get_str_data = $_POST["get_str_data"];
$get_str_model = $_POST["get_str_model"];
$get_str_hy = $_POST["get_str_hy"];


$myfile = fopen("str.txt", "w") or die("Unable to open file!");

fwrite($myfile, $get_str_data);
fwrite($myfile, "\n");
fwrite($myfile, $get_str_model);
fwrite($myfile, "\n");
fwrite($myfile, $get_str_hy);


fclose($myfile);



ds_main($get_str_data,$get_str_model,$get_str_hy);

//$temp = shell_exec('python C:\\xampp\\htdocs\\CNN_Project\\php\\python.py 2>&1');

//print("<div><pre>" . $get_str . "</pre></div>");
$exec = shell_exec('python C:\\xampp\\htdocs\\CNN_Project\\python\\generate_json_tmp.py 2>&1');
echo $exec;

?>