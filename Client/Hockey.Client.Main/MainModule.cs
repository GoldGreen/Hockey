using Hockey.Client.Main.Abstractions;
using Hockey.Client.Main.Connections;
using Hockey.Client.Main.Models;
using Hockey.Client.Main.Services;
using Hockey.Client.Main.ViewModels;
using Hockey.Client.Shared.Prism;
using Prism.Ioc;
using Prism.Modularity;
using Prism.Mvvm;
using Prism.Regions;

namespace Hockey.Client.Main
{
    public class MainModule : IModule
    {
        public void OnInitialized(IContainerProvider containerProvider)
        {
            containerProvider.Resolve<IRegionManager>()
                             .RegisterViewWithRegion<Views.Main>(GlobalRegions.Main);
        }

        public void RegisterTypes(IContainerRegistry containerRegistry)
        {
            containerRegistry.RegisterSingleton<IHockeyService, HockeyService>();
            containerRegistry.RegisterSingleton<IHockeyConnection, HockeyConnection>();
            containerRegistry.RegisterSingleton<IMainModel, MainModel>();

            ViewModelLocationProvider.Register<Views.Main, MainViewModel>();
        }
    }
}
