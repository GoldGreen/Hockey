using System;
using System.Collections.Generic;

namespace Hockey.Client.Shared.Extensions
{
    public static class ObservableExtensions
    {
        private static List<IDisposable> _disposables = new();

        public static IDisposable Cashe(this IDisposable disposable)
        {
            _disposables.Add(disposable);
            return disposable;
        }
    }
}
