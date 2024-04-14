// 单个线程如何利用future来处理多个链接
#include <future>

void process_connections(connection_set &connections) {
  while (!done(connections)) {
    for (connection_iterator connection = connections.begin();
         connection != connections.end(); connection++) {
      if (connection->has_incomming_data()) {
        data_packet data = connection->incomming();
        std::promise<payload_type> &p = connection->get_promise(data.id);
        p.set_value(data.payload);
      }
      if (connection->has_outgoing_data()) {
        outgoing_packet data = connection->top_of_outgoing_queue();
        connection->send(data);
        data.promise.set_value(true);
      }
    }
  }
}
