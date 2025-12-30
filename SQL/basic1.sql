INSERT INTO `user`
(id, pw, nick, created_at)
VALUES
('grara@gmail.com', '122334', '사람인', NOW())

SELECT * FROM `user`
SELECT * FROM `user` WHERE id="grara@gmail.com" AND pw='122334'

UPDATE `user` 
SET 
	pw='newpw' ,
	nick='새로운 닉네임',
	address = '',
	detail_address=''
WHERE idx=1

DELETE FROM `user` WHERE idx=2
