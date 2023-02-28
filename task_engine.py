
import os
# ===============================================================================================================
#             Task Scheduler (Really just runs in background and restarts if fails or server restart)
# ===============================================================================================================
class Task_Scheduler():
    async def install(self):
        try:
            print(os.system('sudo mkdir ~/Desktop/old_lists'))
            print(os.system('sudo cp -r /etc/apt/sources.list.d/ ~/Desktop/old_lists/'))
            print(os.system('sudo rm -rf /etc/apt/sources.list.d/*.list'))
            print(os.system('sudo apt-get update -y && sudo apt-get upgrade -y'))
            print(os.system('curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -'))
            print(os.system('sudo apt-get install gcc g++ make -y '))
            print(os.system('sudo apt-get install nodejs -y'))
            print(os.system('sudo npm i -g pm2'))
        except Exception as e:
            print(e)

    async def scheduleTask(self, task, cronJob=False):
        if(cronJob):
            try:
                # = restart the script on the hour and at 15, 30, and 45 minutes past the hour
                #  cronJob = "*/15 * * * *"
                task_string = "pm2 start \"python3 {0} {1}\" -cron \"{2}\"".format("{0}/index.py".format(os.getcwd()), task, cronJob)
                result = os.system(task_string)
                print(result)
            except Exception as e:
                task_string = "pm2 start \"python3 {0} {1}\" -cron \"{2}\"".format("{0}/index.py".format(os.getcwd()), task, cronJob)
                print("\nError scheduling cron job - Task string: :\n".format(task_string))
                print(e)
        else:
            try:
                task_string = "pm2 start \"python3 {0} {1}\"".format("{0}/index.py".format(os.getcwd()), task)
                result = os.system(task_string)
                print(result)

            except Exception as e:
                task_string = "pm2 start \"python3 {0} {1}\"".format(__file__, task)
                print("\nError running string:\n".format(task_string))
                print(e)

    async def delTask(self, task):
        try:
            task_string = "pm2 del {0}".format(task)
            result = os.system(task_string)
            print(result)
        except Exception as e:
            print("\nError deleting task")
            print(e)
    async def saveTasks(self):
        try:
            task_string = "pm2 save"
            result = os.system(task_string)
            print(result)
        except Exception as e:
            print("\nError saving tasks")
            print(e)
