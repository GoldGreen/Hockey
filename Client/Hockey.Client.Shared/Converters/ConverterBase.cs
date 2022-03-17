using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using System.Windows.Markup;

namespace Hockey.Client.Shared.Converters
{
    public abstract class ConverterBase<T> : MarkupExtension, IValueConverter
        where T : ConverterBase<T>, new()
    {
        private static T _converter;

        public abstract object Convert(object value, Type targetType, object parameter, CultureInfo culture);
        
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }

        public override object ProvideValue(IServiceProvider serviceProvider)
        {
            return _converter ??= new();
        }
    }
}
