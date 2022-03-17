using Microsoft.AspNetCore.SignalR.Client;
using System.Threading.Tasks;

namespace Hockey.Client.Shared.Connections
{
    public class ConnectionBase
    {
        protected readonly HubConnection _connection;

        public ConnectionBase(string address)
        {
            _connection = new HubConnectionBuilder()
                               .WithUrl(address)
                               .Build();
            _ = OnInit();
        }

        protected virtual async Task OnInit()
        {
            await _connection.StartAsync();
        }
    }
}
