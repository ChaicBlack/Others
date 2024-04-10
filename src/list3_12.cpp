/* 这个主要是讲std::call_once和std::once_flag的，用来给接受任务的第一个
 * 线程派发任务的. 以下是通用代码
 * std::shared_ptr<some_resource> resource_ptr;
 * std::once_flag resource_flag;
 * void init_resource(){
 *   resource_ptr.reset(new some_resource);
 * }
 * void foo(){
 *   std::call_once(resource_flag, init_resource);
 *   resource_ptr->do_something();
 * }
 */
// 以下是示例
class X {
private:
  connection_info connection_details;
  connection_handle connetion;
  std::once_flag connection_init_flag;
  void open_connection(){
    connection = connection_manager.open(connection_details);
  }
public:
  X(const connection_info & connection_details_):
    connection_details(connection_details_)
  {}
  void send_data(const data_packet & data){
    std::call_once(connection_init_flag, &X::open_connection, this);
    connection.send_data(data);
  }
  data_packet receive_data(){
    std::call_once(connection_init_flag, &X::open_connection, this);
    return connection.receive_data();
  }
};
