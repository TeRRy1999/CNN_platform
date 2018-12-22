<?php




$stop = "[[0,0]]";

$myfile_val_ls = fopen("C:\\xampp\\htdocs\\CNN_Project\\python\\validation_loss.json", "w") or die("Unable to open file!");
$myfile_val_ac = fopen("C:\\xampp\\htdocs\\CNN_Project\\python\\validation_accuracy.json", "w") or die("Unable to open file!");
$myfile_train_ls = fopen("C:\\xampp\\htdocs\\CNN_Project\\python\\train_loss.json", "w") or die("Unable to open file!");
$myfile_train_ac = fopen("C:\\xampp\\htdocs\\CNN_Project\\python\\train_accuracy.json", "w") or die("Unable to open file!");

fwrite($myfile_val_ls, $stop);
fwrite($myfile_val_ac, $stop);
fwrite($myfile_train_ls, $stop);
fwrite($myfile_train_ac, $stop);


fclose($myfile_val_ls);
fclose($myfile_val_ac);
fclose($myfile_train_ls);
fclose($myfile_train_ac);


?>