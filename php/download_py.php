<?php

$file = "test.py";
$download_path = 'C:\xampp\htdocs\CNN_Project\python\test.py"';
$file_to_download = $download_path; // file to be downloaded

//echo "in php";


if(file_exists($download_path)){
	
	header("Cache-Control: public");
	header("Content-Description: Fil Transfer");
	header("Content-Disposition:attachment; filename=$file");
	header("Content-Transfer-Encoding: binary");

	readfile($file_to_download);
	exit;
}

?>