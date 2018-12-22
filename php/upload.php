<?php

if(isset($_FILES["file"]["type"]))
{
	$validextensions_image = array("jpeg", "jpg", "png", "JPG");
	$validextensions_file = array("zip","rar","7z");
	$temporary = explode(".", $_FILES["file"]["name"]);
	$file_extension = end($temporary);


	if (in_array($file_extension, $validextensions_image) || in_array($file_extension, $validextensions_file)) 
	{
		//echo $temporary[1];
		if ($_FILES["file"]["error"] > 0)
		{
			echo "Return Code: " . $_FILES["file"]["error"] . "<br/><br/>";
		}
		else
		{
			/*if (file_exists("C:/xampp/htdocs/CNN_Project/upload/" . $_FILES["file"]["name"])) {
				echo $_FILES["file"]["name"] . " <span id='invalid'><b>already exists.</b></span> ";
			}
			else
			{*/
				$sourcePath = $_FILES['file']['tmp_name']; // Storing source path of the file in a variable
				if(in_array($file_extension, $validextensions_file))
					$targetPath = "C:/xampp/htdocs/CNN_Project/upload/train_data." . $file_extension;
				else
					$targetPath = "C:/xampp/htdocs/CNN_Project/upload/test.jpg";// . $file_extension; // Target path where file is to be stored
				move_uploaded_file($sourcePath,$targetPath) ; // Moving Uploaded file
				

				echo $filename . " has been uploaded Successfully...!!\n";
				//echo "<br/><b>File Name:</b> " . $_FILES["file"]["name"] . "<br>";
				//echo "<b>Type:</b> " . $_FILES["file"]["type"] . "<br>";
				//echo "<b>Size:</b> " . ($_FILES["file"]["size"] / 1024) . " kB<br>";
				//echo "<b>Temp file:</b> " . $_FILES["file"]["tmp_name"] . "<br>";
			//}
				shell_exec('python C:\\xampp\\htdocs\\CNN_Project\\python\\test.py');
		}
	}
	else
	{
		echo "***Invalid file Size or Type***";
	}
}



?>