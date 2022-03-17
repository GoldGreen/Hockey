using Hockey.Client.Shared.Configuration;
using Prism.Ioc;
using System.Net.Http;

namespace Hockey.Client.Shared.Http
{
    public static class HttpClientExtensions
    {
        public static IContainerRegistry RegisterHttpClient(this IContainerRegistry containerProvider)
        {
            containerProvider.RegisterSingleton<HttpClient>(HttpClientFactory);
            return containerProvider;
        }

        private static HttpClient HttpClientFactory(IContainerProvider containerProvider)
        {
            AddressesOption addresses = containerProvider.Resolve<AddressesOption>();
            return new HttpClient { BaseAddress = new(addresses.Server) };
        }
    }
}
