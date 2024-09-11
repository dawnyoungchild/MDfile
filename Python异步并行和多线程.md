# Python异步并行和多线程

## 1.asyncio异步并行

### 1.1 同步和异步

**同步**：每当系统执行完一段代码或者函数后，系统都将一直等待该段代码或函数的返回值或消息，直到系统接收返回值或消息后才继续执行下一段代码或函数，在等待返回值或消息期间，程序处于阻塞状态，系统将不做任何事情。

**异步**：系统在执行完一段代码或者函数后，不用阻塞性地等待返回值或消息，而是继续执行下一段代码或函数，在同一时间段里执行多个任务（而不是傻傻地等着一件事情做完并且结果出来后才去做下一件事情），将多个任务并行，从而提高程序的执行效率。

**同步和异步都是单线程下的概念。**

### 1.2 协程函数

**协程**：协程是线程的优化，是一种微线程，它是一种比线程更节省资源、效率更高的系统调度机制。异步就是基于协程实现的。

**在Python中，实现协程的模块主要有asyncio、gevent和tornado，使用较多的是asyncio。**

~~~python
# 协程函数的定义和调用

import asyncio
import time

async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')

print(f"程序于{time.strftime('%X')}开始执行")
asyncio.run(main())
print(f"程序于{time.strftime('%X')}执行结束")
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023093009240120230930092401.png" alt="image-20230930092354702" style="zoom:80%;" />

- 在Python中定义函数时，我们通过在def语句前面加上async将函数定义为协程函数

- await asyncio.sleep(1)表示临时中断当前的函数1s。如果程序中还有其他函数，则继续执行下一个函数，直到下一个函数执行完毕，再返回来执行main()函数。因为除了一个main()函数，就没有其他函数了，所以在print('hello')后，main()函数休眠了1s，然后继续print('world')。

- 协程函数不是普通的函数，不能直接用main()来调用，需要使用**asyncio.run(main())**才能执行该协程函数。

**要实现异步并行，需要使用asyncio的create_task()方法将协程函数打包成一个任务**

~~~python
# 如果不将需要运行的函数打包，函数运行时仍然是顺序运行没有并行效果。定义一个say_after函数，并在main函数内调用两次，查看运行时间

import asyncio
import time

async def say_after(what, delay):
    print(what)
    await asyncio.sleep(delay)

async def main():
    print(f"程序于{time.strftime('%X')}开始执行")
    await say_after('hello', 1)
    await say_after('world', 2)
    print(f"程序于{time.strftime('%X')}执行结束")

asyncio.run(main())
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023093009350620230930093506.png" alt="image-20230930093506546" style="zoom:80%;" />

从运行时间上可以看出，两次调用say_after函数并没有并行处理，花费的时间是两次调用的时间之和。

~~~python
# 仍然在main函数中两次调用say_after函数，并使用asyncio的create_task方法将调用的say_after函数打包成一个任务

import asyncio
import time

async def say_after(what, delay):
    print(what)
    await asyncio.sleep(delay)

async def main():
    task1 = asyncio.create_task(say_after('hello', 1))
    task2 = asyncio.create_task(say_after('world', 2))
    print(f"程序于{time.strftime('%X')}开始执行")
    await task1
    await task2
    print(f"程序于{time.strftime('%X')}执行结束")

asyncio.run(main())
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023093009405620230930094056.png" alt="image-20230930094056584" style="zoom:80%;" />

这次运行时间可以看出，两次调用是并行执行的。

**可等待对象**：如果一个对象可以在await语句中使用，那么它就是可等待对象。可等待对象主要有3种类型：协程、任务和Future。

### 总结

- 在Python中，要实现程序异步运行，可以通过将函数定义为协程函数实现，也就是在定义函数时，def语句前面加上async表示这是一个协程函数。
- 使用asyncio的create_task方法，将需要并行的函数打包成任务。
- 使用asyncio.run调用函数，实现异步效果。

~~~python
import asyncio
import time

async def say_after(what, delay):
    print(what)
    await asyncio.sleep(delay)

async def main():
    task1 = asyncio.create_task(say_after('hello', 1))
    task2 = asyncio.create_task(say_after('world', 2))
    print(f"程序于{time.strftime('%X')}开始执行")
    await task1
    await task2
    print(f"程序于{time.strftime('%X')}执行结束")

asyncio.run(main())
~~~

### 举例

在应用时，经常会碰到对列表数据进行操作的情形。如果顺序处理列表元素，用时会比较长。此时可以使用asyncio库进行异步处理

~~~python
# 对列表中的IP地址进行ping测试，并返回测试结果
import asyncio
import ping3

async def ping(ip):
    response_time = ping3.ping(ip)
    if response_time is not None:
        print(f"IP地址 {ip} 的响应时间为 {response_time} 毫秒")
    else:
        print(f"IP地址 {ip} 不可达")

async def main():
    ip_addresses = ["192.168.0.1", "192.168.0.2", "192.168.0.3", ...]  # 替换为你的IP地址列表

    tasks = []
    for ip in ip_addresses:
        tasks.append(ping(ip))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
~~~

以上只举例说明流程，实际操作中ping3模块好像并不支持asyncio异步，也就是说，以上代码与同步代码运行时间是一样的。



## 2.Threading多线程

Python 3已内置了\_thread和threading两个模块来实现多线程。相较于_thread，threading提供的方法更多而且更常用。

~~~python
# 使用Threading模块对函数进行打包，运行查看效果

import threading
import time

def say_after(what, delay):
    print(what)
    time.sleep(delay)
    print(what)

t = threading.Thread(target=say_after, args=('hello',3))

print(f"程序于{time.strftime('%X')}开始执行")
t.start()
print(f"程序于{time.strftime('%X')}结束执行")
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023100116453820231001164538.png" alt="image-20231001164530970" style="zoom:80%;" />

从上例中可以看出，程序中本来有一个3秒的等待时间，但实际运行却是在1秒内完成的。这是因为除了threading.Thread()为say_after()函数创建的**用户线程**，print(f"程序于{time.strftime('%X')}开始执行")和print(f"程序于{time.strftime('%X')}执行结束")两个print()函数也共同占用了**公用的内核线程**。也就是说，该脚本实际上调用了两个线程：一个是用户线程，一个是内核线程，也就构成了一个多线程的环境。因为分属不同的线程，say_after()函数和函数之外的两个print语句是**同时运行的，互不干涉**，所以print(f"程序于{time.strftime('%X')}执行结束")不会像在单线程中那样等到t.start()执行完才被执行，而是在print (f"程序于{time.strftime('%X')} 开始执行")被执行后就马上跟着执行。

如果要让程序顺序运行，需要使用join()方法。**join()方法的作用是强制阻塞调用它的线程，直到该线程运行完毕或者终止（类似单线程同步）。**这时代码就变为：

~~~python
import threading
import time

def say_after(what, delay):
    print(what)
    time.sleep(delay)
    print(what)

t = threading.Thread(target=say_after, args=('hello',3))

print(f"程序于{time.strftime('%X')}开始执行")
t.start()
t.join()	# 变量t调用的start()方法，所以join()方法也要通过变量t调用
print(f"程序于{time.strftime('%X')}结束执行")
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023100116524620231001165246.png" alt="image-20231001165246968" style="zoom:80%;" />

### 举例

这里还是用测试ip地址可达性的例子。在异步处理批量ping测试时，没有能实现节约时间的效果。使用多线程处理ping测试，看一下有没有效果。

~~~python
import threading
import time
import ping3

def ping(ip):
    response_time = ping3.ping(ip)
    if response_time is not None:
        print(f"IP地址 {ip} 的响应时间为 {response_time} 毫秒")
    else:
        print(f"IP地址 {ip} 不可达")

ip_addresses = ["192.168.0.1", "10.253.125.2", "192.168.0.3", "10.253.125.4", "192.168.0.5", "192.168.0.6"]

threads = []
print(f"程序于{time.strftime('%X')}开始执行")
for i in ip_addresses:
    t = threading.Thread(target=ping, args=(i,))	# 此处只有一个参数时，参数后面一点要跟“,”
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()	# 使用join方法确保所有线程执行完成
print(f"程序于{time.strftime('%X')}结束执行")
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2023100117075120231001170751.png" alt="image-20231001170751769" style="zoom:80%;" />

可以看到在使用Theading模块进行ping测试时，返回6个地址的结果仅用时4秒（在asyncio模块下用时是20秒），多线程提速效果明显。