<?php

	if(isset($_POST['upload'])){
		$file = $_FILES['file'];
		print_r($file);
	}
?>
<!DOCTYPE html>
<!DOCTYPE html>
<html>
<head>
	<title>Uploading File</title>
</head>
<body>

	<form action="?" method="POST" enctype="multipart/form-data">
		<label>Upload</label>
		<p><input type="file" name="file"/></p>
		<p><input type="submit" name="upload" value="upload"></p>
		<a href="PDF/1.pdf">Link to a pdf</a>		
	</form>

</body>
</html>