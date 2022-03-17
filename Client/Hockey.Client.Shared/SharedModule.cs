using Hockey.Client.Shared.Configuration;
using Hockey.Client.Shared.Http;
using Prism.Ioc;
using Prism.Modularity;

namespace Hockey.Client.Shared
{
    public class SharedModule : IModule
    {
        public void OnInitialized(IContainerProvider containerProvider)
        {
        }

        public void RegisterTypes(IContainerRegistry containerRegistry)
        {
            containerRegistry.RegisterConfiguration();
            containerRegistry.RegisterOption<AddressesOption>("Addresses");
            containerRegistry.RegisterHttpClient();
        }
    }
}
