# -*- coding: utf-8 -*-
# function: send email
# 'the python file you maybe run' ; python mail.py
import yagmail
import time
import os

password = os.getenv('EMAIL_PASSWORD')

class MailBox():
    def __init__(
        self,
        sender:str = "xxx@vip.163.com",
        receiver:str = "xx@qq.com",
        password:str = "xxxx",
        subject:str = "代码运行完成提醒"
    ):
        self.sender = sender
        self.receiver = receiver
        self.password = password
        self.subject = subject
        self.yag = yagmail.SMTP(user = self.sender, password = self.password, host = 'smtp.qq.com')
    
    def send(self, contents:list):
        self.yag.send(to = self.receiver, subject = self.subject, contents = contents)
        print("邮件发送成功！")
 
# 测试
if __name__ == "__main__":
    mail = MailBox(sender='3557727504@qq.com', 
                   receiver='lizheng.cs@zju.edu.cn', 
                   password=password, 
                   subject='代码运行完成提醒')
    message = "A100服务器上的代码已经运行完成！"
    mail.send([f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n{message}"])